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

# Render #######################################################################

blur_image!(image::Matrix{T}, blur::Float64) where T = begin
  k = Kernel.gaussian(blur)
  # https://github.com/JuliaImages/ImageFiltering.jl/issues/268
  if size(k, 1) < 33
    imfilter!(image, image, k)
  else
    @warn "blur_image! with big blur: $blur"
  end
end

const GlyphImageCache = Vector{Tuple{Glyph,Matrix{RGB24}}}

make_glyph_image_cache(image_size::Tuple{Int,Int}, n_entries::Int) = begin
  [ (Glyph(0, 0, 0, 0, 0, 0.0, 0.0), zeros(RGB24, image_size)) for _ = 1:n_entries ]
end

get_glyph_image(cache::GlyphImageCache, g::Glyph) = begin
  for (kg, img) = cache
    if g == kg
      return img, true
    end
  end
  slot = ceil(Int, rand() * length(cache))
  img = cache[slot][2]
  cache[slot] = (g, img)
  img .= zero(RGB24)
  img, false
end

glyph_image_cache = GlyphImageCache()
n_found = 0
n_notfound = 0

draw_glyph!(image::Matrix{Gray{N0f8}}, g::Glyph) = begin
  global glyph_image_cache, n_found, n_notfound
  if length(glyph_image_cache) == 0 || size(glyph_image_cache[1][2]) != size(image)
    glyph_image_cache = make_glyph_image_cache(size(image), 12)
  end
  glyph_image, found = get_glyph_image(glyph_image_cache, g)
  if !found
    n_notfound += 1
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
    blur_image!(glyph_image, g.blur)
  else
    n_found += 1
  end
  image .= max.(image, red.(glyph_image))  # blit!
end


# Model ########################################################################

const MAX_NUM_GLYPHS = 10

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
  for i = 1:MAX_NUM_GLYPHS
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

show_prior(width, height) = begin
  f = GLMakie.Figure(resolution=(width*4+200, height*3+200))
  ax = [GLMakie.Axis(f[y, x], yreversed = true) for y = 1:3, x = 1:4]
  for x = 1:4, y = 1:3
    tr = simulate(captcha, (width, height))
    image!(ax[y, x], permuteddimsview(get_retval(tr), (2,1)))
  end
  f
end


# Run scaffold #################################################################

if !@isdefined captcha_image
  captcha_image = load("data/captcha.png")
  captcha_image_half = Gray{N0f8}.(restrict(captcha_image))
  captcha_image_quar = Gray{N0f8}.(restrict(captcha_image_half))
  captcha_image_octa = Gray{N0f8}.(restrict(captcha_image_quar))
end

run_inference(image::Matrix{Gray{N0f8}}, inference_step, inference_args) = begin
  height, width = size(image)
  aspect = width/height
  f = GLMakie.Figure(resolution=(2*250*aspect, 250))
  axl = GLMakie.Axis(f[1,1], yreversed = true)
  axr = GLMakie.Axis(f[1,2], yreversed = true)
  image!(axl, permuteddimsview(image, (2,1)))

  obs = choicemap()
  for x = 1:width, y = 1:height  obs[(:image, y, x)] = image[y, x]  end

  scores = Float64[]
  report!(tr) = begin
    image!(axr, permuteddimsview(get_retval(tr), (2,1)))
    push!(scores, get_score(tr))
    print("\riterations: $(length(scores))   score: $(scores[end])")
  end

  tr, = generate(captcha, (width, height), obs)
  report!(tr)
  display(f)

  while f.scene.events.window_open[]
    tr = inference_step(image, tr, report!, f.scene.events.window_open, inference_args)
    report!(tr)
  end
  @label ret
  println()
  println("glyph_cache hit ratio: $(n_found/(n_found + n_notfound))")
  println("trace score: $(get_score(tr))")
  tr, scores
end
