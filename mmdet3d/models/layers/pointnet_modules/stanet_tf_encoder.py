import torch
from torch.nn import MultiheadAttention
import torch.nn as nn
from einops import rearrange, repeat
import torch.nn.functional as F
from einops.layers.torch import Rearrange

class Set_ViT():
    def __init__(self,vocab_size=64,hidden_size=64,
    max_position_embeddings=50,intermediate_size=64,num_hidden_layers=8,
    num_head=8,hidden_dropout_prob=0.1,neighbor_dim=30,radar_dim =6) -> None:
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_head = num_head
        self.hidden_dropout_prob = hidden_dropout_prob
        self.neighbor_dim = neighbor_dim
        self.radar_dim = radar_dim

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
    

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = config.hidden_size
        hidden_dim = config.intermediate_size
        dropout = config.hidden_dropout_prob
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    
    
class TransformerEncoderLayer(nn.Module):
    def __init__(self, config, use_pytorch_version=True):
        super().__init__()
        self.use_pytorch_version = use_pytorch_version
        if use_pytorch_version:
            self.layer_norm_1 = nn.LayerNorm(config.hidden_size)
            self.layer_norm_2 = nn.LayerNorm(config.hidden_size)
            self.attention = MultiheadAttention(config.hidden_size,config.num_head,batch_first = True)
            self.feed_forward = FeedForward(config)
        else:
            dim = config.hidden_size
            heads = config.num_head
            self.layers = nn.ModuleList([])
            for _ in range(config.num_hidden_layers):
                self.layers.append(nn.ModuleList([
                    PreNorm(dim, Attention(dim, heads = heads, dropout = config.hidden_dropout_prob)),
                    PreNorm(dim, FeedForward(config))
                ]))
    def forward(self, x, mask=None):
        # Apply layer normalization and then copy input into query, key, value
        if self.use_pytorch_version:
            if mask is not None:
                b,n = mask.shape
                
                token0_mask = torch.tensor([False]*b).view(-1,1).to(x.device)
                mask = torch.cat((token0_mask,mask),dim=-1)
                
            # Apply attention with a skip connection
            hidden_state = x
            x = x + self.attention(hidden_state, hidden_state, hidden_state, key_padding_mask=mask)[0]
            x = self.layer_norm_1(x)
            #x = x + self.feed_forward(self.layer_norm_2(x))    
            x= self.layer_norm_2(x + self.feed_forward(x))        
        else:
            for attn, ff in self.layers:
                x = attn(x) + x
                x = ff(x) + x
        return x
                
            
class TransformerEncoder(nn.Module):
    def __init__(self, config,use_pytorch_version=True,
                prompt_length=None, embedding_key='cls', prompt_init='uniform', prompt_pool=False, prompt_key=False, pool_size=None,
                top_k=None, batchwise_prompt=False, prompt_key_init='uniform', head_type='token', use_prompt_mask=False,):
        super().__init__()
        self.config = config
        self.embeddings = Embeddings(self.config)
        self.pool = 'cls'
        self.layers = nn.ModuleList([TransformerEncoderLayer(self.config)
                                     for _ in range(self.config.num_hidden_layers)])
        
        if prompt_length is not None and pool_size is not None and prompt_pool:
            self.prompt = Prompt(length=prompt_length, embed_dim=self.config.hidden_size, embedding_key=embedding_key, prompt_init=prompt_init,
                    prompt_pool=prompt_pool, prompt_key=prompt_key, pool_size=pool_size, top_k=top_k, batchwise_prompt=batchwise_prompt,
                    prompt_key_init=prompt_key_init,)
        
        
        
    def forward(self, x, pos,cls_token=None,mask=None):
        x = rearrange(x,'b M n c -> (b M) n c')
        pos = rearrange(pos, 'b M n -> (b M) n')
        
        x = self.embeddings(x,pos,cls_token) # B n c
        for layer in self.layers:
            x = layer(x, mask=mask)

        return x[:,1:],x[:,0] # ..., (b m) c

class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # ___________ design token 0 ________________________
        self.cls_token_default = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.cls_token = nn.Sequential(
            nn.Conv1d(config.radar_dim,32,kernel_size = 1),
            #nn.BatchNorm1d(32), # 627
            nn.LayerNorm([32,config.neighbor_dim]),
            #nn.LeakyReLU(),
            nn.Conv1d(32,self.config.hidden_size,kernel_size=config.neighbor_dim),
        )
        # ___________________________________________________

        self.to_patch_embedding = nn.Sequential(
            # Rearrange('b c m n -> (b m) c n'),
            nn.Conv1d(self.config.vocab_size, self.config.hidden_size,1),
        )
        self.position_embeddings = nn.Embedding(self.config.max_position_embeddings,
                                                self.config.hidden_size)
        self.layer_norm = nn.LayerNorm(self.config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(p=0.1)
        # print(f'self.conf vab:{self.config.vocab_size}')

    def forward(self, input_ids,position_ids,cls_token =None):
        """ Embed the inputs
        Args:
            input_ids B*M,n,c
            position_ids: B*M,1
            cls_token B*M 1 c
        Return:
            B*M n+1 c
        """
        position_ids = position_ids.to(torch.long)
        fixed_time_id = torch.zeros(position_ids.shape[0],1).to(torch.long).to(position_ids.device)
        position_ids = position_ids+1
        position_ids = torch.cat((fixed_time_id,position_ids),dim=1)
        # print(f'position ids{position_ids}')
        position_embeddings = self.position_embeddings(position_ids)

        # Create token and position embeddings
        input_ids = input_ids.permute(0,2,1) # b,c,n
        token_embeddings = self.to_patch_embedding(input_ids).permute(0,2,1) # b,n,c
        
        if cls_token == None:
            cls_tokens = self.cls_token_default.repeat(len(input_ids),1,1) 
        else:
            cls_tokens = self.cls_token(cls_token) 
            cls_tokens = rearrange(cls_tokens,'b c n -> b n c')
        token_embeddings = torch.cat((cls_tokens, token_embeddings), dim=1)
        
        # Combine token and position embeddings
        embeddings = token_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings # B,N,hidden
        

if __name__ == '__main__':
        # embedding_layer = Embeddings(config)
    encoder = TransformerEncoder(config)
    input = torch.rand(2,32,6)
    pos = torch.ones(2,32)
    
    print(encoder(input,pos).shape)
