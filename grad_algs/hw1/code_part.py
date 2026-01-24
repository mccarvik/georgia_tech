
# this is code for the recursion for part 2

dpC[i] = C[i] + min(
    dpC[i-1],                     # stay on Claude (no switch)
    dpG[i-1] + 1                  #switch from Gemma to Claude
)

dpG[i] = G[i] + min(
    dpG[i-1],                     # stay on Gemma (no switch)
    dpC[i-1] + 1                  # switch from Claude to Gemma
)