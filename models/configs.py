import ml_collections
def get_4_config():
    """Returns the ViT-B/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.split = 'non-overlap'
    config.slide_step = 7
    config.hidden_size = 4165
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 12495
    config.transformer.num_heads = 7
    config.transformer.num_layers = 7
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.Liner = 1024
    config.representation_size = None
    return config
def get_3_config():
    """Returns the ViT-B/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.split = 'non-overlap'
    config.slide_step = 7
    config.hidden_size = 1029
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3060
    config.transformer.num_heads = 7
    config.transformer.num_layers = 7
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    config.Liner=768
    config.batch_size = 32
    return config
def get_2_config():
    """Returns the ViT-B/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.split = 'non-overlap'
    config.slide_step = 7
    config.hidden_size = 245
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 735
    config.transformer.num_heads = 7
    config.transformer.num_layers = 7
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.Liner = 512
    config.representation_size = None
    return config
def get_1_config():
    """Returns the ViT-B/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.split = 'non-overlap'
    config.slide_step = 7
    config.hidden_size = 49
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 147
    config.transformer.num_heads = 7
    config.transformer.num_layers = 7
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.Liner = 256
    config.representation_size = None
    return config