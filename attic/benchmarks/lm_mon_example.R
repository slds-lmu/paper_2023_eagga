f = function(x1, x2) {
  y = (3 * x1) + (- 5 * x2) + (10 * sin(x1 * x2))
  y
}

x1 = seq(-10, 10, length.out = 101L)
plot(x1, f(x1, 1), type = "l")

