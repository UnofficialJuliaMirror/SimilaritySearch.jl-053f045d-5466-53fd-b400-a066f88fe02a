#  Copyright 2016 Eric S. Tellez <eric.tellez@infotec.mx>
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http:#www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


### Basic operations on the index
const NNS_PUSH_LOGBASE = 2

function find_neighborhood(index::Graph{T}, item::T) where T
    n::Int = length(index.db)
    n == 0 && return (NnResult(), Int32[])
    k::Int = ceil(Int, log(NNS_PUSH_LOGBASE, 1+n))
    k_1::Int = ceil(Int, log(NNS_PUSH_LOGBASE, 2+n))

    if n > 4 && k != k_1
        optimize!(index, index.recall)
    end

    return neighborhood(index.neighborhood_algo, index, item)
end

function push_neighborhood!(index::Graph{T}, item::T, L::Vector{Int32}, n::Int) where T
    for objID in L
        push!(index.links[objID], 1+n)
    end

    push!(index.links, L)
    push!(index.db, item)
end

function push!(index::Graph{T}, item::T) where T
    knn, neighbors = find_neighborhood(index, item)
    push_neighborhood!(index, item, neighbors, length(index.db))
    index.options.verbose && length(index.db) % 5000 == 0 && info("added n=$(n+1), neighborhood=$(length(neighbors)), $(now())")
    knn, neighbors
end

function fit!(index::Graph{T,D}, dataset::Iter) where {T,D,Iter}
    for item in dataset
        push!(index, item::T)
    end
end

function search(index::Graph{T}, q::T, res::KnnResult{Node}) where T
    search(index.search_algo, index, q, res)
    return res
end

function search(index::Graph{T}, q::T) where T
    return search(index, q, KnnResult(Node, 1))
end

function optimize!(index::Graph{T}, recall::Float64; perf::Nullable{Performance}=Nullable{Performance}()) where T
    perf = isnull(perf) ? Performance(index.db, index.dist) : get(perf)
    optimize_algo!(index.search_algo, index, recall, perf)
end

function search_at(index::Graph{T}, q::T, start::I, res::KnnResult{Node}, tabu) where {T, I<:Integer}
    length(index.db) == 0 && return res

    function oracle(_q)
        index.links[start]
    end

    beam_search(index.search_algo, index, q, res, tabu, Nullable{Function}(oracle))
    return res

    res
end

function search_at(index::Graph{T}, q::T, start::I, res::R) where {T, I <: Integer, R <: Result}
    tabu = falses(length(index.db))
    search_at(index, q, start, res, tabu)
end

function compute_aknn(index::Graph{T}, k::Int) where T # k=index.k, recall=index.recall)
    # optimize!(index, recall, perf=Performance(index.db, index.dist, k=k))
    n = length(index.db)
    aknn = [KnnResult(k) for i=1:n]
    tabu = falses(length(index.db))

    for i=1:n
        if i > 1
            tabu[:] = false
        end

        j = 1

        q = index.db[i]
        res = aknn[i]
        for p in res
            tabu[p.objID] = true
        end

        function oracle(q::T)
            # this can be a very costly operation, or can be a very fast estimator, please do some research about it!!
            if length(res) > 0
                a = Iterators.flatten(index.links[p.objID] for p in res)
                return Iterators.flatten((a, index.links[i]))
            else
                return index.links[i]
            end
        end

        beam_search(index.search_algo, index, q, res, tabu, Nullable{Function}(oracle))
        for p in res
            i < p.objID && push!(aknn[p.objID], i, p.dist)
        end

        if index.options.verbose && (i % 10000) == 1
            info("algorithm=$(index.search_algo), neighborhood_factor=$(index.neighborhood_algo), k=$(k); advance $i of n=$n")
        end
    end

    return aknn
    # newindex = LocalSearchIndex(index.search_algo, index.neighborhood_algo, index.db, index.dist, index.recall, index.k, index.restarts, index.beam_size, index.montecarlo_size, index.candidate_size, aknn)
    # return newindex
end

