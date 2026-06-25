// -*- mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-
// vi: set et ts=4 sw=4 sts=4:
/*
  This file is part of the Open Porous Media project (OPM).

  OPM is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 2 of the License, or
  (at your option) any later version.

  OPM is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with OPM.  If not, see <http://www.gnu.org/licenses/>.
*/
/*!
 * \file
 * \brief Parsing for the experimental, opt-in adaptive-refinement request.
 *
 * This is a deliberately *separate* and *additive* helper for the AdaptiveCpGrid
 * demonstration: it lets the simulator refine a region of a deck that has **no**
 * CARFIN, reusing the exact same static-refinement machinery
 * (CpGrid::addLgrsUpdateLeafView). The request is read from the OPM_ADAPTIVE_LGR
 * environment variable so that nothing in the normal (static-LGR) parameter /
 * deck path is touched. Default (unset) -> the simulator behaves exactly as
 * before.
 */
#ifndef OPM_ADAPTIVE_LGR_HPP
#define OPM_ADAPTIVE_LGR_HPP

#include <array>
#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace Opm {

//! One adaptive refinement box, in the same convention as a CARFIN keyword.
struct AdaptiveLgrBox
{
    std::string name;
    std::array<int,3> startIJK{};   //!< 0-based, inclusive lower corner
    std::array<int,3> endIJK{};     //!< 0-based, exclusive upper corner
    std::array<int,3> cellsPerDim{};//!< subdivisions per parent cell
};

/// Parse an adaptive-refinement spec string. Each box is "I1 I2 J1 J2 K1 K2 NX
/// NY NZ" using the **1-based, inclusive** CARFIN convention (so "5 6 5 6 1 3 6
/// 6 9" is exactly the CARFIN box 'LGR1' 5 6 5 6 1 3 6 6 9). Boxes are separated
/// by ';'. Returns the boxes converted to the half-open 0-based form that
/// CpGrid::addLgrsUpdateLeafView expects. Throws std::invalid_argument on a
/// malformed spec.
inline std::vector<AdaptiveLgrBox> parseAdaptiveLgrSpec(const std::string& spec)
{
    std::vector<AdaptiveLgrBox> boxes;
    std::stringstream boxStream(spec);
    std::string one;
    int idx = 0;
    while (std::getline(boxStream, one, ';')) {
        std::stringstream ss(one);
        std::array<int,6> r{};
        std::array<int,3> nd{};
        bool ok = static_cast<bool>(ss >> r[0] >> r[1] >> r[2] >> r[3] >> r[4] >> r[5]
                                       >> nd[0] >> nd[1] >> nd[2]);
        std::string extra;
        if (!ok || (ss >> extra)) {
            throw std::invalid_argument(
                "OPM_ADAPTIVE_LGR: malformed box '" + one + "'. Expected nine "
                "integers 'I1 I2 J1 J2 K1 K2 NX NY NZ' (1-based, CARFIN style); "
                "separate multiple boxes with ';'.");
        }
        // CARFIN NX/NY/NZ are the *total* number of refined cells across the
        // box, so cells-per-parent-cell = NX / (I2 - I1 + 1) (matching the deck
        // conversion in GenericCpGridVanguard::addLgrsUpdateLeafView). Require an
        // exact division, as CARFIN does.
        const std::array<int,3> nparents{ r[1] - r[0] + 1, r[3] - r[2] + 1, r[5] - r[4] + 1 };
        for (int d = 0; d < 3; ++d) {
            if (nparents[d] <= 0 || nd[d] <= 0 || (nd[d] % nparents[d]) != 0) {
                throw std::invalid_argument(
                    "OPM_ADAPTIVE_LGR: box '" + one + "': the refinement count must "
                    "be a positive multiple of the number of parent cells in each "
                    "direction (CARFIN NX/NY/NZ are box totals).");
            }
        }
        AdaptiveLgrBox b;
        b.name = "ADAPT" + std::to_string(++idx);
        b.startIJK = { r[0] - 1, r[2] - 1, r[4] - 1 };
        b.endIJK   = { r[1],     r[3],     r[5]     };
        b.cellsPerDim = { nd[0] / nparents[0], nd[1] / nparents[1], nd[2] / nparents[2] };
        boxes.push_back(std::move(b));
    }
    return boxes;
}

/// The adaptive-refinement spec from the environment, or "" when unset/blank.
/// Reading an env var (like OPM_LGR_POISON_REFINED) keeps this fully out of the
/// deck/parameter path.
inline std::string adaptiveLgrSpecFromEnv()
{
    const char* env = std::getenv("OPM_ADAPTIVE_LGR");
    if (env == nullptr) {
        return {};
    }
    std::string s{env};
    // blank (all whitespace) counts as unset
    if (s.find_first_not_of(" \t\n") == std::string::npos) {
        return {};
    }
    return s;
}

} // namespace Opm

#endif // OPM_ADAPTIVE_LGR_HPP
