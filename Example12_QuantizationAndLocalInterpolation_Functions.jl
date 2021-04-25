"""
     get_quantizer(n::Int,
                   P::Int,
                   Λ::Array;
                   distance="L2-full")
Input:
 `n::Int`,
  number of of points from the stochastic space.

Output:
 `X::Array{Float64,2}`, `size() = (length(Λ), n)`,
  (non-weighted) coordinates in stochastic space.

 ``
"""
function get_quantizer(n::Int,
                       P::Int,
                       Λ::Array;
                       distance="L2-full")
end

function get_centroidal_preconds(centers::Array{Float64,2},
                                 Λ::Array{Float64,1};
                                 distance="L2-full")

  if distance == "L2-full"

  end
end

function get_clusters_of_preconds(coords::Array{Array{Float64,1},1},
                                           Λ::Array{Float64,1};
                                           distance="L2-full")

  if distance == "L2-full"

  end
end