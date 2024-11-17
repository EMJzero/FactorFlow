from factors import Shape, Coupling

# DIMENSIONS and COUPLING for GEMMS:
# M: Weight/Out rows
# K: Inner dimension, Weight cols/In rows
# N: In/Out cols
gemm_coupling = Coupling(['M', 'K', 'N'], ['K', 'N'], ['M', 'K'], ['M', 'N'])

# DIMENSIONS and COUPLING for CONVOLUTIONS:
# M: Filter num/Out depth
# P: Out height
# Q: Out width
# C: Filter/Input depth
# R: Filter height
# S: Filter width
# => P+R-1: Input height
# => Q+S-1: Input width
conv_coupling = Coupling(['M', 'P', 'Q', 'C', 'R', 'S'], ['C', ['P', 'R'], ['Q', 'S']], ['M', 'C', 'R', 'S'], ['M', 'P', 'Q'])

"""
Generates computation instances for each GEMM of a BERT Transformer
with arbitrary parameters/dimensions. See:
"BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
"""
def comp_BERT(embedding, seq_length, heads, ff_dim):
    assert embedding % heads == 0, f"Embedding dim ({embedding}) must be divisible by the number of heads ({heads})."
    return {
        'KQV': Shape(
            M = embedding*3,
            K = embedding,
            N = seq_length
            ),
        'KTQ': Shape(
            M = seq_length,
            K = embedding//heads,
            N = seq_length
            ),
        'VScores': Shape(
            M = embedding//heads,
            K = seq_length,
            N = seq_length
            ),
        'Out': Shape(
            M = embedding,
            K = embedding,
            N = seq_length
            ),
        'FF1': Shape(
            M = ff_dim,
            K = embedding,
            N = seq_length
            ),
        'FF2': Shape(
            M = embedding,
            K = ff_dim,
            N = seq_length
            )
    }

comp_BERT_base = comp_BERT(768, 1024, 12, 3072)
comp_BERT_large = comp_BERT(1024, 4096, 16, 4096)

comp_harsh_factos_1 = Shape(
    M = 4000,
    K = 6032,
    N = 12000
    )

comp_harsh_factos_2 = Shape(
    M = 7000,
    K = 1440,
    N = 4224
    )

comp_requiring_padding = Shape(
    M = 4037,
    K = 6011,
    N = 12071
    )

"""
GEMMs coming from scientific applications, taken from previous literature:
"Evaluating Spatial Accelerator Architectures with Tiled Matrix-Matrix Multiplication"
"""
comp_maestro_blas = {
    'MB1': Shape(
        M = 8192,
        K = 8192,
        N = 8192
    ),
    'MB2': Shape(
        M = 1024,
        K = 8192,
        N = 1024
    ),
    'MB3': Shape(
        M = 8,
        K = 8192,
        N = 8
    ),
    'MB4': Shape(
        M = 8,
        K = 1024,
        N = 8192
    ),
    'MB5': Shape(
        M = 8192,
        K = 1024,
        N = 8
    ),
    'MB6': Shape(
        M = 512,
        K = 256,
        N = 256
    )
}