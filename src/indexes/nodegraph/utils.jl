
function estimate_knearest{T, D, M}(db::Vector{T}, dist::D, choosek::Int, from::Int, q::T, tabu::M, res::Result, near::Result)
    n::Int32 = length(db)
    # from = max(choosek, from)  ## it has no sense from < choosek!
    nrange = 1:n

    for i in 1:from
        nodeID = rand(nrange)
        @inbounds if !tabu[nodeID]
            d = dist(db[nodeID], q)
            tabu[nodeID] = true
            push!(near, nodeID, d) && push!(res, nodeID, d)
        end
    end

    near
end

#function estimate_knearest{T, D, R, M}(db::Vector{T}, dist::D, choosek::Int, from::Int, q::T, tabu::M, res::R)::KnnResult
#    near = KnnResult(choosek)
#    estimate_knearest(db, dist, choosek, from, q, tabu, res, near)
#end

function estimate_from_oracle{T, D, M}(index::Graph{T,D}, q::T, beam::Result, tabu::M, res::Result, oracle::Function)
    for childID in oracle(q)
        if !tabu[childID]
            tabu[childID] = true
            d = index.dist(index.db[childID], q)
            push!(beam, childID, d) && push!(res, childID, d)
        end
    end
end
