# ttt_config.py

from transformers import PretrainedConfig

class TTTConfig(PretrainedConfig):
    model_type = "ttt"

    def __init__(
        self,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=8,
        mini_batch_size=64,
        ttt_layer_type="mlp",
        pre_conv=True,
        share_qk=True,
        conv_kernel=3,
        use_gate=True,
        ttt_base_lr=0.0001,
        img_size=256,
        patch_size=16,
        in_channels=3,
        num_classes=1,
        layer_norm_eps=1e-6,
        output_hidden_states=False,
        use_cache=False,
        #use_return_dict=True,
        scan_checkpoint_group_size=0,  # Add this attribute
        intermediate_size=1024,  # For SwiGluMLP
        hidden_act="silu",       # For SwiGluMLP
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.mini_batch_size = mini_batch_size
        self.ttt_layer_type = ttt_layer_type
        self.pre_conv = pre_conv
        self.share_qk = share_qk
        self.conv_kernel = conv_kernel
        self.use_gate = use_gate
        self.ttt_base_lr = ttt_base_lr
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_patches = (img_size // patch_size) ** 2
        self.layer_norm_eps = layer_norm_eps
        self.output_hidden_states = output_hidden_states
        self.use_cache = use_cache
        #self.use_return_dict = use_return_dict
        self.scan_checkpoint_group_size = scan_checkpoint_group_size
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
