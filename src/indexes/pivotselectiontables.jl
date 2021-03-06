# This file is a part of SimilaritySearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

export Sss, LaesaTournament #, KvpTournament

function Sss(dist::Function, db::Array{T,1}, alpha::Float64; shuf=false) where T
    pivots = [db[refID] for refID in select_sss(dist, db, alpha, shuf=shuf)]
    fit(Laesa, dist, db, pivots)
end

function LaesaTournament(dist::Function, db::Array{T,1}, numrefs::Int, tournamentsize::Int=3) where T
    pivots = [db[refID] for refID in select_tournament(dist, db, numrefs, tournamentsize)]
    fit(Laesa, dist, db, pivots)
end

# function KvpTournament(db::Array{T,1}, dist::Function, k::Int, numrefs::Int, tournamentsize::Int=0) where T
#     #pivots = [db[refID] for refID in select_tournament(db, numrefs, tournamentsize, dist)]
#     pivots = select_tournament(db, dist, numrefs, tournamentsize)
#     Kvp(db, dist, k, pivots)
# end
