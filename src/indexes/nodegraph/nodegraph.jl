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


module GraphIndex
import ..Result, ..KnnResult, ..Index, ..Performance, ..probe, ..LocalSearchOptions, ..PerformanceResult
import Base: <

abstract type LocalSearchAlgorithm end
abstract type NeighborhoodAlgorithm end

struct NodeID
    id::UInt64
    created::Dates.Millisecond
    
    NodeID() = new(rand(UInt64), now().instant.periods)
end

hash(n::NodeID) = n.id
isequal(a::NodeID, b::NodeID) = (a.id == b.id && a.created == b.created)

# Needed for sorting and searching
function <(a::NodeID, b::NodeID)
    a.created < b.created && return true
    a.created > b.created && return false
    a.id < b.id
end

mutable struct Node{T}
    key::NodeID
    data::T
    forward::Node
    backward::Node
end

Node(data::T) where T = Node(NodeID(), data, Node[], Node[])
hash(node::Node) = hash(node.key)
isequal(a::Node, b::Node) = isequal(a.key, b.key)

mutable struct Graph{T, D}
    nodes::Vector{Node{T}}
    byKey::Dict{NodeID,Node{T}}
    search_algo::LocalSearchAlgorithm
    neighborhood_algo::NeighborhoodAlgorithm
    dist::D
    recall::Float64
    k::Int
    options::LocalSearchOptions
end

function Graph(dtype::Type, dist::D;
               recall=0.9,
               k=10,
               search::LocalSearchAlgorithm=BeamSearch(),
               neighborhood::NeighborhoodAlgorithm=LogSatNeighborhood(1.1),
               options::LocalSearchOptions=LocalSearchOptions()
               ) where D
    Graph(Node{dtype}[], Dict{NodeID,Node{dtype}}(), search, neighborhood, dist, recall, k, options)
end

include("utils.jl")
include("opt.jl")
include("neighborhood/fixedneighborhood.jl")
include("beamsearch.jl")
include("graph.jl")

end
