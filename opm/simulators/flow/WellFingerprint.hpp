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
 * \brief A semantic "before/after" fingerprint of every well and its
 *        connections, for the dynamic-refinement transaction commit-gate.
 *
 * Captured from the Schedule at a given episode BEFORE the world is torn
 * down, compared against the re-derived Schedule of the candidate refined
 * world BEFORE its first nonlinear step. A protected well must come back
 * byte-for-byte on its identity fields (name, connection set + order, level-0
 * Cartesian cell, GLOBAL tag, open/shut state, direction, completion number)
 * and to a tight relative tolerance on CF / Kh / depth. Counting GLOBAL
 * connections alone is not enough -- an added / removed / reordered / retagged
 * connection must be reported precisely.
 */
#ifndef OPM_WELL_FINGERPRINT_HPP
#define OPM_WELL_FINGERPRINT_HPP

#include <opm/input/eclipse/Schedule/Schedule.hpp>
#include <opm/input/eclipse/Schedule/Well/Connection.hpp>
#include <opm/input/eclipse/Schedule/Well/Well.hpp>

#include <opm/common/OpmLog/OpmLog.hpp>

#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include <fmt/format.h>

namespace Opm {

struct WellConnFingerprint
{
    int    complnum{};
    long   globalCart{};
    int    i{}, j{}, k{};
    int    lgrLevel{};      //!< 0 == GLOBAL
    int    state{};         //!< Connection::State as int
    int    dir{};           //!< Connection::Direction as int
    double CF{}, Kh{}, depth{};
};

struct WellFingerprint
{
    std::string name;
    int status{};           //!< Well::Status as int
    std::vector<WellConnFingerprint> conns;   //!< in schedule order
};

//! Capture the fingerprint of every well active at \p episode.
inline std::vector<WellFingerprint>
captureWellFingerprint(const Schedule& schedule, int episode)
{
    std::vector<WellFingerprint> out;
    for (const auto& w : schedule.getWells(static_cast<std::size_t>(episode))) {
        WellFingerprint fp;
        fp.name   = w.name();
        fp.status = static_cast<int>(w.getStatus());
        for (const auto& c : w.getConnections()) {
            WellConnFingerprint cf;
            cf.complnum   = c.complnum();
            cf.globalCart = static_cast<long>(c.global_index());
            cf.i = c.getI(); cf.j = c.getJ(); cf.k = c.getK();
            cf.lgrLevel = c.get_lgr_level();
            cf.state    = static_cast<int>(c.state());
            cf.dir      = static_cast<int>(c.dir());
            cf.CF = c.CF(); cf.Kh = c.Kh(); cf.depth = c.depth();
            fp.conns.push_back(cf);
        }
        out.push_back(std::move(fp));
    }
    return out;
}

//! Compare two fingerprint sets. \p protectedNames (may be empty) are the
//! wells that MUST be identical; any other well is only checked for gross
//! structural change (added/removed connection, GLOBAL->LGR). Returns true
//! iff every required invariant holds; logs every difference.
inline bool
verifyWellFingerprint(const std::vector<WellFingerprint>& before,
                      const std::vector<WellFingerprint>& after,
                      double relTolCFKh = 1e-9,
                      bool throwOnFail = false)
{
    const auto find = [](const std::vector<WellFingerprint>& v, const std::string& n)
        -> const WellFingerprint* {
        for (const auto& f : v) if (f.name == n) return &f;
        return nullptr;
    };
    const auto reltol = [&](double a, double b) {
        const double s = std::max(std::abs(a), std::abs(b));
        return s > 0.0 ? std::abs(a - b) / s <= relTolCFKh : true;
    };

    bool ok = true;
    std::string msg = "well fingerprint (candidate vs pre-rebuild):";

    for (const auto& b : before) {
        const auto* a = find(after, b.name);
        if (!a) {
            msg += fmt::format("\n  {}: MISSING after rebuild", b.name);
            ok = false;
            continue;
        }
        if (a->status != b.status)
            { msg += fmt::format("\n  {}: status {} -> {}", b.name, b.status, a->status); ok = false; }
        if (a->conns.size() != b.conns.size()) {
            msg += fmt::format("\n  {}: connection count {} -> {}",
                               b.name, b.conns.size(), a->conns.size());
            ok = false;
            continue;
        }
        for (std::size_t c = 0; c < b.conns.size(); ++c) {
            const auto& bc = b.conns[c];
            const auto& ac = a->conns[c];
            std::string d;
            if (ac.complnum   != bc.complnum)   d += fmt::format(" complnum {}->{}", bc.complnum, ac.complnum);
            if (ac.globalCart != bc.globalCart) d += fmt::format(" cell {}->{}", bc.globalCart, ac.globalCart);
            if (ac.i != bc.i || ac.j != bc.j || ac.k != bc.k)
                d += fmt::format(" ijk ({},{},{})->({},{},{})", bc.i, bc.j, bc.k, ac.i, ac.j, ac.k);
            if (ac.lgrLevel   != bc.lgrLevel)   d += fmt::format(" lgr {}->{}", bc.lgrLevel, ac.lgrLevel);
            if (ac.state      != bc.state)      d += fmt::format(" state {}->{}", bc.state, ac.state);
            if (ac.dir        != bc.dir)        d += fmt::format(" dir {}->{}", bc.dir, ac.dir);
            if (!reltol(ac.CF, bc.CF))          d += fmt::format(" CF {:.6e}->{:.6e}", bc.CF, ac.CF);
            if (!reltol(ac.Kh, bc.Kh))          d += fmt::format(" Kh {:.6e}->{:.6e}", bc.Kh, ac.Kh);
            if (!reltol(ac.depth, bc.depth))    d += fmt::format(" depth {:.4f}->{:.4f}", bc.depth, ac.depth);
            if (!d.empty()) { msg += fmt::format("\n  {} conn[{}]:{}", b.name, c, d); ok = false; }
        }
    }
    for (const auto& a : after)
        if (!find(before, a.name)) {
            msg += fmt::format("\n  {}: APPEARED after rebuild", a.name);
            ok = false;
        }

    if (ok) OpmLog::info(msg + "\n  -> all wells preserved");
    else {
        OpmLog::error(msg + "\n  -> well state NOT preserved across the rebuild");
        if (throwOnFail)
            throw std::runtime_error("dynamic refinement: well fingerprint changed");
    }
    return ok;
}

} // namespace Opm

#endif // OPM_WELL_FINGERPRINT_HPP
