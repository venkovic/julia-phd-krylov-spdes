function cov_sexp(x1::Float64,
                  y1::Float64,
                  x2::Float64, 
                  y2::Float64,
                  sig2::Float64,
                  L::Float64)

  return sig2 * exp(-((x1 - x2)^ 2 + (y1 - y2)^2) / L^2)
end