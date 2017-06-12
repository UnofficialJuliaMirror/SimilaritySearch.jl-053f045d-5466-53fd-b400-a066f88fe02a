#  Copyright 2016 Eric S. Tellez <eric.tellez@infotec.mx>
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

export BeamSearch

struct BeamSearch <: LocalSearchAlgorithm
    candidates_size::Int
    montecarlo_size::Int
    beam_size::Int
end

BeamSearch() = BeamSearch(1, 1, 1)
BeamSearch(other::BeamSearch) = BeamSearch(other.candidates_size, other.montecarlo_size, other.beam_size)

### local search algorithm

function beam_search(bsearch::BeamSearch, index::Graph{T}, q::T, res::Result, tabu, oracle::Nullable{Function}) where T
    # first beam
    beam = KnnResult(Node, bsearch.beam_size)

    if isnull(oracle)
        estimate_knearest(index.db, index.dist, bsearch.candidates_size, bsearch.montecarlo_size, q, tabu, res, beam)
    else
        estimate_from_oracle(index, q, beam, tabu, res, get(oracle))
    end

    new_beam = KnnResult(Node, bsearch.beam_size)
    # new_beam = KnnResult(bsearch.beam_size)
    # cov::Float32 = -1.0
    # while cov != last(res).dist
    @inbounds while length(beam) > 0
        clear!(new_beam)
        cov = last(res).dist
        for node in beam
            for nearNode in node.forward
                if nearNode in tabu
                    push!(tabu, nearNode)

                    d = index.dist(nearNode.data, q)
                    if d <= cov
                        push!(new_beam, nearNode, d) && push!(res, nearNode, d)
                    end
                end
            end
        end

        beam, new_beam = new_beam, beam
    end

    beam
end

function search(bsearch::BeamSearch, index::Graph{T}, q::T, res::KnnResult{Node}) where T
    length(index.db) == 0 && return res
    
    tabu = Set{Node}()
    beam_search(bsearch, index, q, res, tabu, index.options.oracle)
    return res
end

function opt_create_random_state(algo::BeamSearch, max_value)
    a = max(1, rand() * max_value |> round |> Int)
    b = max(1, rand() * max_value |> round |> Int)
    c = max(1, rand() * max_value |> round |> Int)
    return BeamSearch(a, b, c)
end

function opt_expand_neighborhood(fun, gsearch::BeamSearch, n::Int, iter::Int)
    f(x, w) = max(1, x + w)
    g(x) = max(1, x + ceil(Int, (rand()-0.5) * log2(n)))

    if iter == 1
        for i in 1:8
            opt_create_random_state(gsearch, ceil(Int, log2(n))) |> fun
        end

        BeamSearch(gsearch.candidates_size |> g, gsearch.montecarlo_size |> g, gsearch.beam_size |> g) |> fun
        BeamSearch(gsearch.candidates_size |> g, gsearch.montecarlo_size, gsearch.beam_size) |> fun
        BeamSearch(gsearch.candidates_size, gsearch.montecarlo_size |> g, gsearch.beam_size) |> fun
        BeamSearch(gsearch.candidates_size, gsearch.montecarlo_size, gsearch.beam_size |> g) |> fun
    end

    w = 2
    while w <= div(32,iter)
        BeamSearch(f(gsearch.candidates_size,  w), gsearch.montecarlo_size, gsearch.beam_size) |> fun
        BeamSearch(f(gsearch.candidates_size, -w), gsearch.montecarlo_size, gsearch.beam_size) |> fun
        BeamSearch(gsearch.candidates_size, f(gsearch.montecarlo_size,  w), gsearch.beam_size) |> fun
        BeamSearch(gsearch.candidates_size, f(gsearch.montecarlo_size, -w), gsearch.beam_size) |> fun
        BeamSearch(gsearch.candidates_size, gsearch.montecarlo_size, f(gsearch.beam_size,  w)) |> fun
        BeamSearch(gsearch.candidates_size, gsearch.montecarlo_size, f(gsearch.beam_size, -w)) |> fun
        w += w
    end
end
