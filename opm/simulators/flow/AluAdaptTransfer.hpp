// -*- mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-
// vi: set et ts=4 sw=4 sts=4:
/*
  This file is part of the Open Porous Media project (OPM).

  OPM is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
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
 * \brief Prolong-by-injection transfer of a per-leaf-cell quantity across a
 *        single in-place grid adaptation, keyed by the grid's persistent local
 *        id (serial only).
 */
#ifndef OPM_ALU_ADAPT_TRANSFER_HPP
#define OPM_ALU_ADAPT_TRANSFER_HPP

#include <dune/grid/common/mcmgmapper.hh>

#include <cstddef>
#include <functional>
#include <unordered_map>
#include <vector>

namespace Opm {

/*!
 * \brief Snapshot a per-leaf quantity before grid.adapt(), keyed by persistent
 *        local id, so it can be re-scattered afterwards.
 *
 * \tparam GridView  leaf grid view type
 * \tparam T         the per-cell value type (must be copy-assignable)
 */
template <class GridView, class T>
class AluAdaptSnapshot
{
    using Grid = std::decay_t<decltype(std::declval<GridView>().grid())>;
    using IdSet = typename Grid::LocalIdSet;
    using IdType = typename IdSet::IdType;

public:
    //! Capture values(leafIndex) for every current interior leaf cell.
    template <class Accessor>
    void capture(const GridView& gv, Accessor&& value)
    {
        map_.clear();
        const auto& idSet = gv.grid().localIdSet();
        Dune::MultipleCodimMultipleGeomTypeMapper<GridView> mapper(gv, Dune::mcmgElementLayout());
        for (const auto& e : elements(gv)) {
            map_.emplace(idSet.id(e), value(mapper.index(e)));
        }
    }

    /*!
     * \brief After adapt(): return the value for a leaf cell -- its own snapshot
     *        if it persisted, otherwise its nearest ancestor's (prolongation by
     *        injection). Throws if no ancestor is found (should not happen for a
     *        pure refinement).
     */
    template <class Entity>
    const T& lookup(const GridView& gv, const Entity& leaf) const
    {
        const auto& idSet = gv.grid().localIdSet();
        auto a = leaf;
        while (true) {
            auto it = map_.find(idSet.id(a));
            if (it != map_.end()) {
                return it->second;
            }
            if (a.level() == 0 || !a.hasFather()) {
                break;
            }
            a = a.father();
        }
        throw std::runtime_error("AluAdaptSnapshot::lookup: no pre-adapt ancestor "
                                 "for a leaf cell -- transfer would lose data.");
    }

    std::size_t size() const { return map_.size(); }

private:
    std::unordered_map<IdType, T> map_;
};

} // namespace Opm

#endif // OPM_ALU_ADAPT_TRANSFER_HPP
