import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock , VAE_ResidualBlock

#conv2d structure
#Input → Conv → BatchNorm → Activation (ReLU) → (optional Dropout)

#VAE_ResidualBlock structure
#it has additional skip connection which is +Input here 
#Input → Conv → BN → ReLU → Conv → BN → +Input → ReLU → Output

'''
       ┌─────────┐
       │         │
       │         ▼
x ──> Conv → BN → ReLU → Conv → BN ──> + ──> ReLU → y
       ▲                         │
       └─────────skip───────────┘

'''

#encoder = we increse the channel size and decrese the size of the image 
class VAE_Encoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            #(Batch_size,channel,height,width) --> (Batch_size, 128, height, width)
            nn.Conv2d(3, 128, kernel_size=3, padding=1),

            #(Batch_size,128,height,width) --> (Batch_size, 128, height, width)
            VAE_ResidualBlock(128,128),

            #(Batch_size,128,height,width) --> (Batch_size, 128, height, width)
            VAE_ResidualBlock(128,128),

            #(Batch_size,128,height,width) --> (Batch_size, 128, height/2, width/2)
            nn.Conv2d(128, 128, kernel_size=3, stride = 2, padding=0),

            #(Batch_size,128,height,width) --> (Batch_size, 256, height/2, width/2)
            VAE_ResidualBlock(128,256),

            #(Batch_size,256,height,width) --> (Batch_size, 256, height/2, width/2)
            VAE_ResidualBlock(256,256),

            #(Batch_size,256,height,width) -->(Batch_size,256,height/2,width/2)
            nn.Conv2d(256,256, kernel_size=3, stride=2, padding =0),

            #(batch_size,256,height,width)--?(batch_size,512,height/8,width/8)
            VAE_ResidualBlock(256,512),

            #(batch_size,512,height,width)--?(batch_size,512,height/8,width/8)
            VAE_ResidualBlock(512,512),

            #(batch_size,512,height,width)-->(batch_size,512,height/8,width/8)
            nn.Conv2d(512,512, kernel_size=3, stride=2, padding=0),

            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),

            #---------now the attentionblock
            #(batch_size,512,height,width)-->(batch_size,512,height/8,width/8)
            VAE_AttentionBlock(512),

            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),

            #(batch_size,512,height,width)-->(batch_size,512,height/8,width/8)
            #groupnormalization
            nn.GroupNorm(32, 512,),

            #(batch_size,512,height,width)-->(batch_size,512,height/8,width/8)
           
            nn.SilU(),
            
            #(batch_size,512,height,width)-->(batch_size,8,height/8,width/8)
            nn.Conv2d(512, 8, kernel_size=3, padding=1),

            #(batch_size,512,height,width)-->(batch_size,512,height/8,width/8)
            nn.Conv2d(8, 8, kernel_size=1, padding=0)
        )

    def forward(self, x: torch.Tensor, noise: torch.Tensor) ->torch.Tensor:
        # x: (Batch_size,channel,height,width)
        #output of last layer is the size of the noise
        #noise (Batch_size, Out_channels, Height/8, Width/8)

        for module in self:
            if getattr(module, 'stride', None) == (2,2):
                #basically for the conv2d and VAE's that have no padding
                #(Padding_left, Padding_right, Padding_Top, Padding_bottom)
                x = F.pad(x,(0, 1, 0, 1))
            x= module(x)

        #the output of variational encoder is mean and variance 
        #(batch_size,8,height/8,width/8)-->two tensors of shape (batch_size,4,height/8,width/8)
        '''
        encoder output qϕ​(z∣x)=N(z;μ(x),σ2(x)I)
        mu is the mean sigma is the variance 
        to train we need parametrs of phi, we cannot directly sample it 
        z∼N(μ,σ2)
    
        '''
        
        mean, log_variance = torch.chunk(x,2, dim=1)

        #(batch_size,8,height/8,width/8)-->two tensors of shape (batch_size,4,height/8,width/8)
        #clamp here makes the log_variance between values -30 and 20
        #log variance for stability as variance cannot be < than 0 or explode kaboom 
        log_variance = torch.clamp(log_variance, -30,20)

        #(batch_size,4,height/8,width/8)--> (batch_size,4,height/8,width/8)
        variance = log_variance.exp()

        #(batch_size,4,height/8,width/8)--> (batch_size,4,height/8,width/8)
        stdev = vaiance.sqrt()

        #math here
        #Sampling called reparameterization trick 
        '''z=μ+σ⊙ϵ
        where:
        ϵ∼N(0,I) (pure noise, independent of μ and σ)
        ⊙ is elementwise multiplication
        '''
        # Z=N(0,1) -> N(mean,variance)=X?
        # X= mean + stdev * noise
      
        #scale the output by a constant
        x *=0.18215

        return

        



