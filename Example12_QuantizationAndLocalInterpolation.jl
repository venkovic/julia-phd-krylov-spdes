using Clustering

d, n = 10, 100

k = 10

X = rand(d, n)

res1 = kmeans(X, k1)
res1.assignments
res1.centers



X, centers, assignments, cost, = get_quantizer(n, P, Λ, distance="L2-full")