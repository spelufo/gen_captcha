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
  "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
]

draw_glyph!(image::Matrix{Gray{N0f8}}, g::Glyph) = begin
  height, width = size(image)
  glyph_image = zeros(RGB24, size(image))
  cr = CairoContext(CairoImageSurface(glyph_image))
  # Transform the canvas to match cairo and julia images coord systems.
  rotate(cr, π/2); translate(cr, 0.0, -height)
  set_source_rgb(cr, 1.0, 1.0, 1.0)
  select_font_face(cr, "Sans", Cairo.FONT_SLANT_NORMAL, Cairo.FONT_WEIGHT_NORMAL)
  set_font_size(cr, g.size_x)  # TODO: size_y.
  move_to(cr, g.pos_x - g.size_x/2, g.pos_y + g.size_y/2)
  show_text(cr, GLYPHS[g.id])
  # Debug rectangle.
  # set_line_width(cr, 2)
  # rectangle(cr, 10, 10, width - 20, height - 20)
  # stroke(cr)
  imfilter!(glyph_image, glyph_image, Kernel.gaussian(g.blur))
  image .= max.(image, red.(glyph_image))  # blit!
end

blur_image!(image::Matrix{Gray{N0f8}}, blur::Float64) = begin
  imfilter!(image, image, Kernel.gaussian(blur))
end

@gen glyph(width::Int, height::Int) = begin
  pos_x = round(Int16, {:pos_x} ~ normal(width/2, width/4))
  pos_y = round(Int16, {:pos_y} ~ normal(height/2, height/4))
  size_x ~ uniform_discrete(50, 100)
  size_y = size_x
  # size_y ~ uniform_discrete(50, 100)
  # rotation ~ uniform(-20.0, 20.0)
  rotation = 0
  id ~ uniform_discrete(1, length(GLYPHS))
  blur = 7 * ({:blur_over_7} ~ beta(1, 2))
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
  global_blur = 7 * ({:global_blur_over_7} ~ beta(1, 2))
  blur_image!(image, global_blur)

  # Noise model.
  epsilon ~ gamma(1, 1)
  for x = 1:width, y = 1:height
    {(:image, y, x)} ~ normal(Float64(image[y, x]), epsilon)
  end

  image
end

run(image::Matrix{Gray{N0f8}}) = begin
  height, width = size(image)
  f = GLMakie.Figure(resolution=(width*2 + 100, height + 100))
  axl = GLMakie.Axis(f[1,1])
  axr = GLMakie.Axis(f[1,2])
  image!(axl, permuteddimsview(image, (2,1)))

  obs = choicemap()
  for x = 1:width, y = 1:height
    obs[(:image, y, x)] = image[y, x]
  end

  tr, = generate(captcha, (width, height), obs)
  image!(axr, permuteddimsview(get_retval(tr), (2,1)))
  display(f)

  iter = 1
  while f.scene.events.window_open[]
    i = ceil(Int, rand() * 10)
    for j = 1:30
      tr, _ = Gen.mh(tr, Gen.select((:glyph, i) => :id))
      tr, _ = Gen.mh(tr, Gen.select((:glyph, i) => :pos_x, (:glyph, i) => :pos_y))
      tr, _ = Gen.mh(tr, Gen.select((:glyph, i) => :size_x))
      tr, _ = Gen.mh(tr, Gen.select((:glyph, i) => :blur_over_7))

      image!(axr, permuteddimsview(get_retval(tr), (2,1)))
      print("\riterations: $iter        ")
      iter += 1
      if !f.scene.events.window_open[]  return tr  end
    end
    tr, _ = Gen.mh(tr, Gen.select(:global_blur))
    # tr, _ = Gen.update(tr, obs) # This should be redundant, I think.

    image!(axr, permuteddimsview(get_retval(tr), (2,1)))
    print("\riterations: $iter (outer)")
    iter += 1
  end

  tr
end

show_prior(width, height) = begin
  f = GLMakie.Figure(resolution=(width*4+200, height*3+200))
  ax = [GLMakie.Axis(f[y, x]) for y = 1:3, x = 1:4]
  for x = 1:4, y = 1:3
    tr = simulate(captcha, (width, height))
    image!(ax[y, x], permuteddimsview(get_retval(tr), (2,1)))
  end
  f
end

if !@isdefined data
  data = load("captcha.png")
end
