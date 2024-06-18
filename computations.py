from factors import Shape

def comp_BERT(embedding, seq_length, heads, ff_dim):
    assert embedding % heads == 0, f"Embedding dim ({embedding}) must be divisible by the number of heads ({heads})."
    return {
        'KQV': Shape(
            D = embedding*3,
            E = embedding,
            L = seq_length
            ),
        'KTQ': Shape(
            D = seq_length,
            E = embedding//heads,
            L = seq_length
            ),
        'VScores': Shape(
            D = embedding//heads,
            E = seq_length,
            L = seq_length
            ),
        'Out': Shape(
            D = embedding,
            E = embedding,
            L = seq_length
            ),
        'FF1': Shape(
            D = ff_dim,
            E = embedding,
            L = seq_length
            ),
        'FF2': Shape(
            D = embedding,
            E = ff_dim,
            L = seq_length
            )
    }

comp_BERT_base = comp_BERT(768, 1024, 12, 3072)
comp_BERT_large = comp_BERT(1024, 4096, 16, 4096)

comp_harsh_factos = Shape(
    D = 4000,
    E = 6032,
    L = 12000
    )

comp_requiring_padding = Shape(
    D = 4037,
    E = 6011,
    L = 12071
    )

comp_maestro_blas = [
    Shape(
        D = 8192,
        E = 8192,
        L = 8192
    ), Shape(
        D = 1024,
        E = 8192,
        L = 1024
    ), Shape(
        D = 8,
        E = 8192,
        L = 8
    ), Shape(
        D = 8,
        E = 1024,
        L = 8192
    ), Shape(
        D = 8192,
        E = 1024,
        L = 8
    ), Shape(
        D = 512,
        E = 256,
        L = 256
    )
]