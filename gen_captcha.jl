using Cairo, Images, ImageFiltering, GLMakie
using Gen

struct Glyph
  id :: UInt8
  pos_x :: Int16
  pos_y :: Int16
  size_x :: Int16
  size_y :: Int16
  rotation :: Float64
  blur :: Float64
end

const GLYPHS = [
  "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
  "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
  "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
  "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
  "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
]

glyph_image = nothing

draw_glyph!(image::Matrix{Gray{N0f8}}, g::Glyph) = begin
  global glyph_image
  if isnothing(glyph_image) || size(glyph_image) != size(image)
    glyph_image = zeros(RGB24, size(image))
  else
    glyph_image .= zero(RGB24)
  end
  cr = CairoContext(CairoImageSurface(glyph_image))
  # Transform the canvas to match cairo and julia images coord systems.
  rotate(cr, π/2); scale(cr, 1.0, -1.0)
  set_source_rgb(cr, 1.0, 1.0, 1.0)
  select_font_face(cr, "Sans", Cairo.FONT_SLANT_NORMAL, Cairo.FONT_WEIGHT_NORMAL)
  set_font_size(cr, g.size_x)  # TODO: size_y.
  move_to(cr, g.pos_x - g.size_x/2, g.pos_y + g.size_y/2)
  show_text(cr, GLYPHS[g.id])
  # Debug rectangle.
  # set_line_width(cr, 2); rectangle(cr, 10, 10, width - 20, height - 20); stroke(cr)
  imfilter!(glyph_image, glyph_image, Kernel.gaussian(g.blur))
  image .= max.(image, red.(glyph_image))  # blit!
end

blur_image!(image::Matrix{Gray{N0f8}}, blur::Float64) = begin
  imfilter!(image, image, Kernel.gaussian(blur))
end

@dist blur_beta(width, a, b) =
  max(1, width÷28) * beta(a, b)

@gen glyph(width::Int, height::Int) = begin
  pos_x = round(Int16, {:pos_x} ~ normal(width/2, width/4))
  pos_y = round(Int16, {:pos_y} ~ normal(height/2, height/4))
  size_x ~ uniform_discrete(width÷12, width÷3)
  size_y = size_x  # TODO: size_y ~ uniform_discrete(width÷6, width÷2)
  rotation = 0     # TODO: rotation ~ uniform(-20.0, 20.0)
  id ~ uniform_discrete(1, length(GLYPHS))
  blur ~ blur_beta(width, 1, 2)
  Glyph(id, pos_x, pos_y, size_x, size_y, rotation, blur)
end


@gen captcha(width::Int, height::Int) = begin
  # Generate image.
  image = zeros(Gray{N0f8}, height, width)
  max_num_glyphs = 10
  for i = 1:max_num_glyphs
    if ({(:is_present, i)} ~ bernoulli(0.5))
      g = {(:glyph, i)} ~ glyph(width, height)
      draw_glyph!(image, g)
    end
  end
  global_blur ~ blur_beta(width, 1, 2)
  blur_image!(image, global_blur)

  # Noise model.
  epsilon ~ gamma(1, 1)
  for x = 1:width, y = 1:height
    {(:image, y, x)} ~ normal(Float64(image[y, x]), epsilon)
  end

  image
end

# Proposals

make_glyph_match_probs() = begin
  nglyphs = length(GLYPHS)
  S = zeros(Float64, nglyphs, nglyphs)
  image_i = zeros(Gray{N0f8}, 50, 50)
  image_j = zeros(Gray{N0f8}, 50, 50)
  nglyphs = length(GLYPHS)
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

# Inference

run(image::Matrix{Gray{N0f8}}) = begin
  height, width = size(image)
  f = GLMakie.Figure(resolution=(width*2 + 100, height + 100))
  axl = GLMakie.Axis(f[1,1], yreversed = true)
  axr = GLMakie.Axis(f[1,2], yreversed = true)
  image!(axl, permuteddimsview(image, (2,1)))

  glyph_transition_probs = make_glyph_match_probs()

  obs = choicemap()
  for x = 1:width, y = 1:height
    obs[(:image, y, x)] = image[y, x]
  end

  tr, = Gen.generate(captcha, (width, height), obs)
  # tr, = Gen.importance_resampling(captcha, (width, height), obs, 20)
  image!(axr, permuteddimsview(get_retval(tr), (2,1)))
  display(f)

  iter = 1
  while f.scene.events.window_open[]
    i = ceil(Int, rand() * 10)
    tr, _ = Gen.mh(tr, Gen.select((:is_present, i)))
    if tr[(:is_present, i)]
      for j = 1:3
        for k = 1:3
          iter += 1
          image!(axr, permuteddimsview(get_retval(tr), (2,1)))
          print("\riterations: $iter        ")
          tr, _ = Gen.mh(tr, random_glyph_step, (i, glyph_transition_probs))
        end
        tr, _ = Gen.mh(tr, random_walk, ((:glyph, i) => :pos_x, 5.0))
        tr, _ = Gen.mh(tr, random_walk, ((:glyph, i) => :pos_y, 5.0))
        tr, _ = Gen.mh(tr, random_walk_discrete, ((:glyph, i) => :size_x, 5))
        tr, _ = Gen.mh(tr, Gen.select((:glyph, i) => :blur))

        if !f.scene.events.window_open[]  return tr  end
      end
    end 
    tr, _ = Gen.mh(tr, Gen.select(:global_blur))
    tr, _ = Gen.mh(tr, Gen.select(:epsilon))

    image!(axr, permuteddimsview(get_retval(tr), (2,1)))
    print("\riterations: $iter (outer)")
    iter += 1
  end

  tr
end

show_prior(width, height) = begin
  f = GLMakie.Figure(resolution=(width*4+200, height*3+200))
  ax = [GLMakie.Axis(f[y, x], yreversed = true) for y = 1:3, x = 1:4]
  for x = 1:4, y = 1:3
    tr = simulate(captcha, (width, height))
    image!(ax[y, x], permuteddimsview(get_retval(tr), (2,1)))
  end
  f
end

if !@isdefined captcha_image
  captcha_image = load("captcha_3.png")
  captcha_image_half = Gray{N0f8}.(restrict(captcha_image))
  captcha_image_quar = Gray{N0f8}.(restrict(captcha_image_half))
  captcha_image_octa = Gray{N0f8}.(restrict(captcha_image_quar))
end

# julia> include("gen_captcha.jl"); trace = run(captcha_image);

