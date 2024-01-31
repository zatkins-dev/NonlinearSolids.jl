function ensurevec(x)
  return x isa AbstractArray ? vec(x) : vec([x])
end
