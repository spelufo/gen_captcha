include("gen_captcha.jl")


inference_step(image, tr, report!, keep_going, args) = begin
  i = ceil(Int, rand() * 10)
  tr, _ = mh(tr, select((:is_present, i)))
  if tr[(:is_present, i)]
    for k = 1:10
      tr, _ = mh(tr, select((:glyph, i) => :id))
    end
    tr, _ = mh(tr, select((:glyph, i) => :pos_x, (:glyph, i) => :pos_y))
    tr, _ = mh(tr, select((:glyph, i) => :size_x))
    tr, _ = mh(tr, select((:glyph, i) => :blur))

    report!(tr)
    if !keep_going[]  return tr  end
  end
  tr, _ = mh(tr, select(:epsilon, :global_blur))
  tr
end

run(image::Matrix{Gray{N0f8}}) = begin
  run_inference(image, inference_step, nothing)
end

