"""
Comprehensive tests for UNet model architectures.
This test suite demonstrates testing best practices for deep learning models.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch

from models.unet import UNet
from models.attention_unet import AttentionUNet, AttentionGate, SpatialAttention, ChannelAttention


class TestUNet:
    """Test suite for basic UNet model."""
    
    @pytest.fixture
    def sample_input(self):
        """Create sample input tensor."""
        return torch.randn(2, 3, 256, 256)
    
    @pytest.fixture
    def unet_model(self):
        """Create UNet model instance."""
        return UNet(n_channels=3, n_classes=1, bilinear=False)
    
    def test_unet_initialization(self, unet_model):
        """Test UNet model initialization."""
        assert isinstance(unet_model, UNet)
        assert unet_model.n_channels == 3
        assert unet_model.n_classes == 1
        assert unet_model.bilinear == False
        
    def test_unet_forward_pass(self, unet_model, sample_input):
        """Test UNet forward pass."""
        output = unet_model(sample_input)
        
        # Check output shape
        expected_shape = (2, 1, 256, 256)
        assert output.shape == expected_shape
        
        # Check output type
        assert isinstance(output, torch.Tensor)
        
    def test_unet_bilinear_mode(self):
        """Test UNet with bilinear upsampling."""
        model = UNet(n_channels=3, n_classes=1, bilinear=True)
        sample_input = torch.randn(1, 3, 256, 256)
        output = model(sample_input)
        
        assert output.shape == (1, 1, 256, 256)
        
    def test_unet_different_input_sizes(self, unet_model):
        """Test UNet with different input sizes."""
        sizes = [(128, 128), (256, 256), (512, 512)]
        
        for height, width in sizes:
            sample_input = torch.randn(1, 3, height, width)
            output = unet_model(sample_input)
            assert output.shape == (1, 1, height, width)
            
    def test_unet_parameter_count(self, unet_model):
        """Test UNet parameter count."""
        total_params = sum(p.numel() for p in unet_model.parameters())
        trainable_params = sum(p.numel() for p in unet_model.parameters() if p.requires_grad)
        
        # UNet should have reasonable number of parameters
        assert total_params > 1000000  # Should have >1M parameters
        assert trainable_params == total_params  # All parameters should be trainable
        
    def test_unet_gradient_flow(self, unet_model, sample_input):
        """Test gradient flow through UNet."""
        output = unet_model(sample_input)
        loss = output.mean()
        loss.backward()
        
        # Check that gradients are computed
        for name, param in unet_model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"
                
    def test_unet_device_transfer(self, unet_model):
        """Test UNet device transfer."""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            model_cuda = unet_model.to(device)
            sample_input = torch.randn(1, 3, 256, 256).to(device)
            
            output = model_cuda(sample_input)
            assert output.device == device
            
    def test_unet_output_range(self, unet_model, sample_input):
        """Test UNet output range (should be logits)."""
        output = unet_model(sample_input)
        
        # Output should be logits (unbounded)
        assert torch.isfinite(output).all()
        # No specific range constraint for logits
        
    def test_unet_memory_efficiency(self, unet_model, sample_input):
        """Test UNet memory efficiency."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()
            
            model_cuda = unet_model.cuda()
            sample_input = sample_input.cuda()
            
            output = model_cuda(sample_input)
            final_memory = torch.cuda.memory_allocated()
            
            # Memory usage should be reasonable
            memory_used = final_memory - initial_memory
            assert memory_used < 2 * 1024 * 1024 * 1024  # Less than 2GB


class TestAttentionUNet:
    """Test suite for Attention UNet model."""
    
    @pytest.fixture
    def sample_input(self):
        """Create sample input tensor."""
        return torch.randn(2, 3, 256, 256)
    
    @pytest.fixture
    def attention_unet_model(self):
        """Create Attention UNet model instance."""
        return AttentionUNet(
            n_channels=3, 
            n_classes=1, 
            bilinear=False,
            use_attention=True,
            use_spatial_attention=True,
            use_channel_attention=True,
            dropout_rate=0.1
        )
    
    def test_attention_unet_initialization(self, attention_unet_model):
        """Test Attention UNet model initialization."""
        assert isinstance(attention_unet_model, AttentionUNet)
        assert attention_unet_model.n_channels == 3
        assert attention_unet_model.n_classes == 1
        assert attention_unet_model.use_attention == True
        assert attention_unet_model.use_spatial_attention == True
        assert attention_unet_model.use_channel_attention == True
        
    def test_attention_unet_forward_pass(self, attention_unet_model, sample_input):
        """Test Attention UNet forward pass."""
        output = attention_unet_model(sample_input)
        
        # Check output shape
        expected_shape = (2, 1, 256, 256)
        assert output.shape == expected_shape
        
        # Check output type
        assert isinstance(output, torch.Tensor)
        
    def test_attention_unet_without_attention(self):
        """Test Attention UNet without attention mechanisms."""
        model = AttentionUNet(
            n_channels=3, 
            n_classes=1, 
            bilinear=False,
            use_attention=False,
            use_spatial_attention=False,
            use_channel_attention=False
        )
        
        sample_input = torch.randn(1, 3, 256, 256)
        output = model(sample_input)
        
        assert output.shape == (1, 1, 256, 256)
        
    def test_attention_unet_dropout(self):
        """Test Attention UNet with dropout."""
        model = AttentionUNet(
            n_channels=3, 
            n_classes=1, 
            dropout_rate=0.5
        )
        
        sample_input = torch.randn(1, 3, 256, 256)
        
        # Test in training mode
        model.train()
        output_train = model(sample_input)
        
        # Test in eval mode
        model.eval()
        output_eval = model(sample_input)
        
        # Outputs should be different due to dropout
        assert not torch.allclose(output_train, output_eval)
        
    def test_attention_unet_parameter_count(self, attention_unet_model):
        """Test Attention UNet parameter count."""
        total_params = sum(p.numel() for p in attention_unet_model.parameters())
        trainable_params = sum(p.numel() for p in attention_unet_model.parameters() if p.requires_grad)
        
        # Attention UNet should have more parameters than basic UNet
        assert total_params > 1500000  # Should have >1.5M parameters
        assert trainable_params == total_params
        
    def test_attention_unet_gradient_flow(self, attention_unet_model, sample_input):
        """Test gradient flow through Attention UNet."""
        output = attention_unet_model(sample_input)
        loss = output.mean()
        loss.backward()
        
        # Check that gradients are computed
        for name, param in attention_unet_model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"


class TestAttentionGate:
    """Test suite for Attention Gate module."""
    
    @pytest.fixture
    def attention_gate(self):
        """Create Attention Gate instance."""
        return AttentionGate(F_g=64, F_l=64, F_int=32)
    
    def test_attention_gate_initialization(self, attention_gate):
        """Test Attention Gate initialization."""
        assert isinstance(attention_gate, AttentionGate)
        
    def test_attention_gate_forward_pass(self, attention_gate):
        """Test Attention Gate forward pass."""
        g = torch.randn(1, 64, 32, 32)  # Gating signal
        x = torch.randn(1, 64, 32, 32)  # Local features
        
        output = attention_gate(g, x)
        
        # Check output shape
        assert output.shape == x.shape
        
        # Check output range (should be weighted input)
        assert torch.isfinite(output).all()
        
    def test_attention_gate_attention_weights(self, attention_gate):
        """Test that attention gate produces meaningful attention weights."""
        g = torch.randn(1, 64, 32, 32)
        x = torch.randn(1, 64, 32, 32)
        
        # Get attention weights
        g1 = attention_gate.W_g(g)
        x1 = attention_gate.W_x(x)
        psi = attention_gate.relu(g1 + x1)
        attention_weights = attention_gate.psi(psi)
        
        # Attention weights should be between 0 and 1
        assert torch.all(attention_weights >= 0)
        assert torch.all(attention_weights <= 1)


class TestSpatialAttention:
    """Test suite for Spatial Attention module."""
    
    @pytest.fixture
    def spatial_attention(self):
        """Create Spatial Attention instance."""
        return SpatialAttention(kernel_size=7)
    
    def test_spatial_attention_initialization(self, spatial_attention):
        """Test Spatial Attention initialization."""
        assert isinstance(spatial_attention, SpatialAttention)
        
    def test_spatial_attention_forward_pass(self, spatial_attention):
        """Test Spatial Attention forward pass."""
        x = torch.randn(1, 64, 32, 32)
        output = spatial_attention(x)
        
        # Check output shape
        assert output.shape == x.shape
        
        # Check output range
        assert torch.isfinite(output).all()
        
    def test_spatial_attention_attention_map(self, spatial_attention):
        """Test that spatial attention produces meaningful attention maps."""
        x = torch.randn(1, 64, 32, 32)
        
        # Compute attention map manually
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention_input = torch.cat([avg_out, max_out], dim=1)
        attention_map = spatial_attention.sigmoid(spatial_attention.conv(attention_input))
        
        # Attention map should be between 0 and 1
        assert torch.all(attention_map >= 0)
        assert torch.all(attention_map <= 1)


class TestChannelAttention:
    """Test suite for Channel Attention module."""
    
    @pytest.fixture
    def channel_attention(self):
        """Create Channel Attention instance."""
        return ChannelAttention(in_channels=64, reduction=16)
    
    def test_channel_attention_initialization(self, channel_attention):
        """Test Channel Attention initialization."""
        assert isinstance(channel_attention, ChannelAttention)
        
    def test_channel_attention_forward_pass(self, channel_attention):
        """Test Channel Attention forward pass."""
        x = torch.randn(1, 64, 32, 32)
        output = channel_attention(x)
        
        # Check output shape
        assert output.shape == x.shape
        
        # Check output range
        assert torch.isfinite(output).all()
        
    def test_channel_attention_attention_weights(self, channel_attention):
        """Test that channel attention produces meaningful attention weights."""
        x = torch.randn(1, 64, 32, 32)
        
        # Compute attention weights manually
        avg_out = channel_attention.fc(channel_attention.avg_pool(x))
        max_out = channel_attention.fc(channel_attention.max_pool(x))
        attention_weights = channel_attention.sigmoid(avg_out + max_out)
        
        # Attention weights should be between 0 and 1
        assert torch.all(attention_weights >= 0)
        assert torch.all(attention_weights <= 1)


class TestModelIntegration:
    """Integration tests for model components."""
    
    def test_model_output_consistency(self):
        """Test that model outputs are consistent across runs."""
        model = UNet(n_channels=3, n_classes=1)
        sample_input = torch.randn(1, 3, 256, 256)
        
        # Set model to eval mode for consistency
        model.eval()
        
        with torch.no_grad():
            output1 = model(sample_input)
            output2 = model(sample_input)
            
        # Outputs should be identical
        assert torch.allclose(output1, output2)
        
    def test_model_parameter_sharing(self):
        """Test that model parameters are properly shared."""
        model = AttentionUNet(n_channels=3, n_classes=1)
        
        # Get parameter names
        param_names = list(model.state_dict().keys())
        
        # Check that all parameters are unique
        assert len(param_names) == len(set(param_names))
        
    def test_model_save_load(self):
        """Test model save and load functionality."""
        model = UNet(n_channels=3, n_classes=1)
        sample_input = torch.randn(1, 3, 256, 256)
        
        # Get original output
        model.eval()
        with torch.no_grad():
            original_output = model(sample_input)
        
        # Save model
        torch.save(model.state_dict(), 'test_model.pth')
        
        # Load model
        new_model = UNet(n_channels=3, n_classes=1)
        new_model.load_state_dict(torch.load('test_model.pth'))
        new_model.eval()
        
        # Get new output
        with torch.no_grad():
            new_output = new_model(sample_input)
        
        # Outputs should be identical
        assert torch.allclose(original_output, new_output)
        
        # Clean up
        import os
        os.remove('test_model.pth')


class TestModelPerformance:
    """Performance tests for models."""
    
    def test_model_inference_time(self):
        """Test model inference time."""
        model = UNet(n_channels=3, n_classes=1)
        sample_input = torch.randn(1, 3, 256, 256)
        
        model.eval()
        
        # Warm up
        with torch.no_grad():
            for _ in range(10):
                _ = model(sample_input)
        
        # Measure inference time
        import time
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(100):
                _ = model(sample_input)
        
        end_time = time.time()
        avg_time = (end_time - start_time) / 100
        
        # Inference should be reasonably fast
        assert avg_time < 0.1  # Less than 100ms per inference
        
    def test_model_memory_usage(self):
        """Test model memory usage."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()
            
            model = UNet(n_channels=3, n_classes=1).cuda()
            sample_input = torch.randn(1, 3, 256, 256).cuda()
            
            # Forward pass
            output = model(sample_input)
            
            final_memory = torch.cuda.memory_allocated()
            memory_used = final_memory - initial_memory
            
            # Memory usage should be reasonable
            assert memory_used < 1024 * 1024 * 1024  # Less than 1GB


if __name__ == "__main__":
    pytest.main([__file__]) 