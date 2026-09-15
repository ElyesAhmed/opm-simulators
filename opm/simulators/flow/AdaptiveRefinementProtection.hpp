/*
  This file is part of the Open Porous Media project (OPM).

  OPM is free software: you can redistribute it and/or modify it under the
  terms of the GNU General Public License as published by the Free Software
  Foundation, either version 3 of the License, or (at your option) any later
  version.

  OPM is distributed in the hope that it will be useful, but WITHOUT ANY
  WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
  FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
  details.

  You should have received a copy of the GNU General Public License along
  with OPM.  If not, see <http://www.gnu.org/licenses/>.
*/
/*!
 * \file
 * \brief ONE source of truth for the "cells a generated refinement box must
 *        never contain" set, shared by the a posteriori estimator (which
 *        excludes them from Dorfler marking) and by the dynamic-refinement
 *        driver's preflight (which rejects a spec that enters them).
 *
 * The set is the union, over EVERY schedule snapshot, of:
 *   - every well-completion cell (a coarse completion must stay GLOBAL);
 *   - every possible future connection cell (ACTIONX etc.);
 *   - for a VERTICAL well (all its connections share one (i,j)), the whole
 *     spanned k-interval OF THAT WELL -- so a box cannot bridge the trajectory
 *     between two completions of the same well. Two independent wells (or a
 *     well and a SOURCE) that happen to share an (i,j) column do NOT create a
 *     joint interval;
 *   - every SOURCE-keyword cell (a point flux singularity, EXACT cell only --
 *     never a column interval);
 *   - optionally, a \p haloRings-cell dilation of all of the above (a
 *     near-singularity coarse ring; 0 = off).
 *
 * Indices are GLOBAL Cartesian: (k*ny + j)*nx + i.
 */
#ifndef OPM_ADAPTIVE_REFINEMENT_PROTECTION_HPP
#define OPM_ADAPTIVE_REFINEMENT_PROTECTION_HPP

#include <opm/input/eclipse/Schedule/Schedule.hpp>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace Opm {

inline int refinementProtectionHaloFromEnvironment()
{
    const char* text = std::getenv("OPM_APOST_PROTECT_HALO");
    if (text == nullptr) {
        return 0;
    }

    int value = 0;
    char trailing = '\0';
    std::istringstream input(text);
    if (!(input >> value) || (input >> trailing) || value < 0) {
        throw std::invalid_argument(
            "OPM_APOST_PROTECT_HALO must be a non-negative integer");
    }
    return value;
}

inline std::vector<int>
buildProtectedRefinementCells(const Schedule& schedule,
                              const std::array<int, 3>& dims,
                              int haloRings = 0)
{
    const std::int64_t nx = dims[0], ny = dims[1], nz = dims[2];
    const std::int64_t ncart = nx * ny * nz;
    const auto lin = [nx, ny](std::int64_t i, std::int64_t j, std::int64_t k) {
        return (k * ny + j) * nx + i;
    };

    std::vector<char> mask(static_cast<std::size_t>(std::max<std::int64_t>(ncart, 0)), 0);
    const auto set = [&](std::int64_t c) {
        if (c >= 0 && c < ncart) mask[static_cast<std::size_t>(c)] = 1;
    };

    // Per well, collect EVERY completion cell it ever uses (all snapshots) PLUS
    // its possible future ACTIONX connections, as (i,j,k). Then decide the
    // vertical k-interval from that merged set -- two future connections of one
    // vertical well no longer leave the cells between them unprotected
    // (reviews 2026-09-10).
    const auto future = schedule.getPossibleFutureConnections();
    std::unordered_map<std::string, std::vector<std::array<int, 3>>> wellCells;
    for (std::size_t step = 0; step < schedule.size(); ++step) {
        for (const auto& well : schedule.getWells(step)) {
            auto& v = wellCells[well.name()];
            for (const auto& c : well.getConnections()) {
                set(static_cast<std::int64_t>(c.global_index()));   // exact, always
                v.push_back({c.getI(), c.getJ(), c.getK()});
            }
        }
        for (const auto& [ijk, cells] : schedule[step].source()) {
            static_cast<void>(cells);
            set(lin(ijk[0], ijk[1], ijk[2]));   // EXACT source cell only, never a column
        }
    }
    for (const auto& [wname, cells] : future) {
        auto& v = wellCells[wname];
        for (const auto gi : cells) {
            set(static_cast<std::int64_t>(gi));   // exact
            const long g = static_cast<long>(gi);
            v.push_back({static_cast<int>(g % nx),
                         static_cast<int>((g / nx) % ny),
                         static_cast<int>(g / (nx * ny))});
        }
    }
    for (const auto& [wname, v] : wellCells) {
        static_cast<void>(wname);
        if (v.empty()) continue;
        const int wi = v[0][0], wj = v[0][1];
        bool vertical = true;
        int kmin = v[0][2], kmax = v[0][2];
        for (const auto& c : v) {
            if (c[0] != wi || c[1] != wj) { vertical = false; break; }
            kmin = std::min(kmin, c[2]);
            kmax = std::max(kmax, c[2]);
        }
        // straight vertical well -> protect its whole spanned k-interval;
        // deviated / multi-column -> exact cells only (already set above).
        if (vertical)
            for (int k = kmin; k <= kmax; ++k)
                set(lin(wi, wj, k));
    }

    // near-singularity halo (boolean-mask dilation, no duplicate growth)
    for (int pass = 0; pass < std::max(0, haloRings); ++pass) {
        std::vector<char> grown = mask;
        for (std::int64_t k = 0; k < nz; ++k)
        for (std::int64_t j = 0; j < ny; ++j)
        for (std::int64_t i = 0; i < nx; ++i) {
            if (!mask[static_cast<std::size_t>(lin(i, j, k))]) continue;
            for (const auto& d : {std::array<std::int64_t, 3>{1, 0, 0}, {-1, 0, 0},
                                  {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}}) {
                const std::int64_t ni = i + d[0], nj = j + d[1], nk = k + d[2];
                if (ni < 0 || nj < 0 || nk < 0 || ni >= nx || nj >= ny || nk >= nz) continue;
                grown[static_cast<std::size_t>(lin(ni, nj, nk))] = 1;
            }
        }
        mask.swap(grown);
    }

    std::vector<int> out;
    for (std::int64_t c = 0; c < ncart; ++c)
        if (mask[static_cast<std::size_t>(c)]) out.push_back(static_cast<int>(c));
    return out;   // ascending by construction
}

} // namespace Opm

#endif // OPM_ADAPTIVE_REFINEMENT_PROTECTION_HPP
