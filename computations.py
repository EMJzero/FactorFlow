from factors import Shape

def comp_BERT(embedding, seq_length, heads, ff_dim):
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

comp_BERT_base = comp_BERT(768, 1024, 15, 3072)
comp_BERT_large = comp_BERT(1024, 4096, 16, 4096)