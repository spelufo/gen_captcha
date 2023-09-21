include("gen_captcha.jl")


# Proposals

make_glyph_match_probs() = begin
  nglyphs = length(GLYPHS)
  S = zeros(Float64, nglyphs, nglyphs)
  image_i = zeros(Gray{N0f8}, 50, 50)
  image_j = zeros(Gray{N0f8}, 50, 50)
  for i= 1:nglyphs, j = i:nglyphs
    image_i .= zero(Gray{N0f8})
    image_j .= zero(Gray{N0f8})
    draw_glyph!(image_i, Glyph(i, 25, 25, 30, 30, 0, 0.1))
    draw_glyph!(image_j, Glyph(j, 25, 25, 30, 30, 0, 0.1))
    # S[i, j] = S[j, i] = sum(image_i .>= 0.1 .&& image_j .>= 0.1)
    S[i, j] = S[j, i] = sum(image_i .== image_j)
  end
  for i = 1:nglyphs
    S[:, i] .= S[:, i] ./ sum(S[:, i])
  end
  S
end

@gen random_glyph_step(tr, i, glyph_transition_probs) = begin
  glyph_id = tr[(:glyph, i) => :id]
  {(:glyph, i) => :id} ~ categorical(glyph_transition_probs[:, glyph_id])
end

@gen random_walk_discrete(tr, addr, step) = begin
  {addr} ~ uniform_discrete(tr[addr]-step, tr[addr]+step)
end

@gen random_walk(tr, addr, step) = begin
  {addr} ~ normal(tr[addr], step)
end

# I can't seem to make this work with a @gen proposal. If I use non-traced rand
# then I get KeyErrors because the proposal is run twice (fwd, bwd) and different
# keys are used each time.
# Even when I don't, I think the dependance on the trace data with the if makes
# it so that different addresses are visited on the fwd / bwd executions, and I
# get "Did not visit all constraints" errors.
# It seems like the way to do this is to do trace translation / involution as in
# the reversible jump MCMC tutorial. I'll have to try that.
@gen random_merge_close_glyphs(tr, width, height) = begin
  # candidates = Tuple{Int,Int}[]
  # bestd, besti, bestj = width^2 + height^2, 1, 2
  close = zeros(Bool)
  for i = 1:MAX_NUM_GLYPHS
    if !tr[(:is_present, i)] continue end
    xi = tr[(:glyph, i) => :pos_x]; yi = tr[(:glyph, i) => :pos_y]
    for j = i+1:MAX_NUM_GLYPHS
      if !tr[(:is_present, j)] continue end
      xj = tr[(:glyph, j) => :pos_x]; yj = tr[(:glyph, j) => :pos_y]
      d = (yi - yj)^2 + (xi - xj)^2
      if d < width/10
        {(:is_present, i)} ~ bernoulli(1.0)
        wi = tr[(:glyph, i) => :size_x]
        # {(:glyph, j)} ~ glyph(width, height)
        avg_x = (tr[(:glyph, i) => :pos_x] + tr[(:glyph, j) => :pos_x])/2
        avg_y = (tr[(:glyph, i) => :pos_y] + tr[(:glyph, j) => :pos_y])/2
        {(:glyph, i) => :pos_x} ~ normal(avg_x, width/16)
        {(:glyph, i) => :pos_y} ~ normal(avg_y, width/16)
        {(:glyph, i) => :size_x} ~ uniform_discrete(min(wi, wj), wi + wj)
        {(:glyph, i) => :id} ~ uniform_discrete(1, length(GLYPHS))
        {(:glyph, i) => :blur} ~ blur_beta(width, 1, 2)
      else
        {(:is_present, j)} ~ bernoulli(0.5)
        {(:is_present, i)} ~ bernoulli(0.5)
        {(:glyph, i)} ~ glyph(width, height)
        {(:glyph, j)} ~ glyph(width, height)
      end
    end
  end
  nothing
end

# Inference

inference_step(image, tr, report!, keep_going, glyph_transition_probs) = begin
  i = ceil(Int, rand() * MAX_NUM_GLYPHS)
  tr, _ = mh(tr, select((:is_present, i)))
  if tr[(:is_present, i)]
    for k = 1:10
      tr, _ = mh(tr, random_glyph_step, (i, glyph_transition_probs))
      report!(tr)
    end
    tr, _ = mh(tr, random_walk, ((:glyph, i) => :pos_x, 10.0))
    tr, _ = mh(tr, random_walk, ((:glyph, i) => :pos_y, 10.0))
    tr, _ = mh(tr, random_walk_discrete, ((:glyph, i) => :size_x, 2))
    tr, _ = mh(tr, random_walk, ((:glyph, i) => :blur, 2.0))

    report!(tr)
    if !keep_going[]  return tr  end
  end 
  tr, _ = mh(tr, select(:epsilon, :global_blur))
  tr
end

run(image::Matrix{Gray{N0f8}}) = begin
  glyph_transition_probs = make_glyph_match_probs()
  run_inference(image, inference_step, glyph_transition_probs)
end

