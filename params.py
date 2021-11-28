


# Parameters
img_size = 224
patch_size = 16

# Encoder
encoder_dim = 192
encoder_depth = 12
encoder_heads = 3

# Decoder
decoder_dim = 512
decoder_depth = 8
decoder_heads = 16


# Masking phase
# 1. We use the shuffle patch after Sin-Cos position embeeding for encoder.
# 2. Mask the shuffle patch, keep the mask index.
# 3. Unshuffle the mask patch and combine with the encoder embeeding before the position embeeding 
#    for decoder.
# 4. Restruction decoder embeeidng by convtranspose.
# 5.Build the mask map with mask index for cal the loss(only consider the mask patch).
