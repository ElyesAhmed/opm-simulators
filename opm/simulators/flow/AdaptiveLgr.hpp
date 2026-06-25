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
 * \brief Parameter + parsing for the adaptive-refinement request used by the
 *        flow_blackoil_adaptive executable (AdaptiveCpGridVanguard).
 *
 * It lets that executable refine a region of a deck that has **no** CARFIN,
 * reusing the same static-refinement machinery (CpGrid::addLgrsUpdateLeafView).
 * This is entirely separate from the standard flow executables / Vanguard.
 */
#ifndef OPM_ADAPTIVE_LGR_HPP
#define OPM_ADAPTIVE_LGR_HPP

#include <array>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace Opm::Parameters {

//! \brief Adaptive refinement boxes for flow_blackoil_adaptive (--adaptive-lgr).
//! CARFIN-style "I1 I2 J1 J2 K1 K2 NX NY NZ" (1-based), ';'-separated boxes.
struct AdaptiveLgr { static constexpr auto value = ""; };

} // namespace Opm::Parameters

namespace Opm {

//! One adaptive refinement box, in the same convention as a CARFIN keyword.
struct AdaptiveLgrBox
{
    std::string name;
    std::array<int,3> startIJK{};   //!< 0-based, inclusive lower corner
    std::array<int,3> endIJK{};     //!< 0-based, exclusive upper corner
    std::array<int,3> cellsPerDim{};//!< subdivisions per parent cell
};

/// Parse an adaptive-refinement spec. Each box is "I1 I2 J1 J2 K1 K2 NX NY NZ"
/// in the **1-based, inclusive** CARFIN convention (so "5 6 5 6 1 3 6 6 9" is
/// exactly the CARFIN box 5 6 5 6 1 3 6 6 9). NX/NY/NZ are box totals, so the
/// per-parent-cell subdivision is NX/(I2-I1+1) etc. -- matching the deck
/// conversion in GenericCpGridVanguard::addLgrsUpdateLeafView. Boxes separated by
/// ';'. Throws std::invalid_argument on a malformed spec.
inline std::vector<AdaptiveLgrBox> parseAdaptiveLgrSpec(const std::string& spec)
{
    std::vector<AdaptiveLgrBox> boxes;
    std::stringstream boxStream(spec);
    std::string one;
    int idx = 0;
    while (std::getline(boxStream, one, ';')) {
        if (one.find_first_not_of(" \t") == std::string::npos) {
            continue; // skip blank segments
        }
        std::stringstream ss(one);
        std::array<int,6> r{};
        std::array<int,3> nd{};
        const bool ok = static_cast<bool>(
            ss >> r[0] >> r[1] >> r[2] >> r[3] >> r[4] >> r[5] >> nd[0] >> nd[1] >> nd[2]);
        std::string extra;
        if (!ok || (ss >> extra)) {
            throw std::invalid_argument(
                "--adaptive-lgr: malformed box '" + one + "'. Expected nine integers "
                "'I1 I2 J1 J2 K1 K2 NX NY NZ' (1-based, CARFIN style); separate "
                "multiple boxes with ';'.");
        }
        const std::array<int,3> nparents{ r[1] - r[0] + 1, r[3] - r[2] + 1, r[5] - r[4] + 1 };
        for (int d = 0; d < 3; ++d) {
            if (nparents[d] <= 0 || nd[d] <= 0 || (nd[d] % nparents[d]) != 0) {
                throw std::invalid_argument(
                    "--adaptive-lgr: box '" + one + "': the refinement count must be a "
                    "positive multiple of the number of parent cells per direction "
                    "(CARFIN NX/NY/NZ are box totals).");
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

} // namespace Opm

#endif // OPM_ADAPTIVE_LGR_HPP
