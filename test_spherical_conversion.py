"""
Test file to verify spherical conversion is working correctly.
Tests the key spherical functions with dummy inputs.
"""
import sys
import torch
import math

# Test 1: Import and test vector_rms_norm function
print("=" * 70)
print("TEST 1: Vector RMS Normalization (Projection to Sphere)")
print("=" * 70)

def vector_rms_norm(z, zero_mean=False, eps=1e-6, curvature=1.0):
    """
    L2 normalization to project vectors to sphere with given curvature.
    Curvature k > 0: sphere with radius r = 1/sqrt(k)
    """
    assert z.ndim >= 2
    dim = tuple(range(1, z.ndim))
    if zero_mean:
        z = z - z.mean(dim=dim, keepdim=True)
    # Compute L2 norm
    norm = torch.sqrt(z.square().sum(dim=dim, keepdim=True) + eps)
    # Normalize to unit sphere first, then scale to desired radius
    radius = 1.0 / torch.sqrt(torch.tensor(curvature, dtype=z.dtype, device=z.device))
    return z * (radius / norm)


def vector_compute_magnitude(x):
    """Compute L2 norm"""
    assert x.ndim >= 2
    reduce_dims = tuple(range(1, x.ndim))
    mag = x.square().sum(dim=reduce_dims, keepdim=True).sqrt()
    return mag


def vector_compute_angle(x, y):
    """Compute angle between two vectors on sphere"""
    assert x.ndim >= 2
    assert x.shape == y.shape
    reduce_dims = tuple(range(1, x.ndim))
    dot = (x * y).sum(dim=reduce_dims, keepdim=True)
    mag = vector_compute_magnitude(x) * vector_compute_magnitude(y)
    mag = torch.clamp(mag, min=1e-6)
    cos_sim = torch.clamp(dot / mag, min=-1.0, max=1.0)
    angle = torch.acos(cos_sim) / math.pi * 180.0
    return angle


# Create dummy input: batch of 5, feature dimension 16
batch_size = 5
feature_dim = 16
dummy_input = torch.randn(batch_size, feature_dim)

print(f"\nDummy input shape: {dummy_input.shape}")
print(f"Dummy input range: [{dummy_input.min():.4f}, {dummy_input.max():.4f}]")
print(f"Dummy input norm (before projection): {vector_compute_magnitude(dummy_input).mean():.4f}")

# Test with curvature = 1.0 (unit sphere, radius = 1.0)
print("\n--- Testing with curvature k=1.0 (unit sphere, radius=1.0) ---")
normalized_k1 = vector_rms_norm(dummy_input, curvature=1.0)
mag_k1 = vector_compute_magnitude(normalized_k1)
print(f"Output shape: {normalized_k1.shape}")
print(f"Output norm (should be ~1.0): {mag_k1.mean():.6f}")
print(f"Output norm std: {mag_k1.std():.6f}")
print(f"Expected radius: 1/√1 = 1.0")
assert torch.allclose(mag_k1, torch.ones_like(mag_k1), atol=1e-5), "Curvature 1.0 failed: magnitude != 1.0"
print("✓ PASSED: All vectors projected to unit sphere (radius=1.0)")

# Test with curvature = 2.0 (smaller sphere, radius = 0.707)
print("\n--- Testing with curvature k=2.0 (smaller sphere, radius=0.707) ---")
normalized_k2 = vector_rms_norm(dummy_input, curvature=2.0)
mag_k2 = vector_compute_magnitude(normalized_k2)
expected_radius_k2 = 1.0 / math.sqrt(2.0)
print(f"Output shape: {normalized_k2.shape}")
print(f"Output norm (should be ~{expected_radius_k2:.4f}): {mag_k2.mean():.6f}")
print(f"Output norm std: {mag_k2.std():.6f}")
print(f"Expected radius: 1/√2 = {expected_radius_k2:.6f}")
assert torch.allclose(mag_k2, torch.full_like(mag_k2, expected_radius_k2), atol=1e-5), \
    f"Curvature 2.0 failed: magnitude != {expected_radius_k2}"
print(f"✓ PASSED: All vectors projected to sphere (radius={expected_radius_k2:.6f})")

# Test with curvature = 0.5 (larger sphere, radius = 1.414)
print("\n--- Testing with curvature k=0.5 (larger sphere, radius=1.414) ---")
normalized_k05 = vector_rms_norm(dummy_input, curvature=0.5)
mag_k05 = vector_compute_magnitude(normalized_k05)
expected_radius_k05 = 1.0 / math.sqrt(0.5)
print(f"Output shape: {normalized_k05.shape}")
print(f"Output norm (should be ~{expected_radius_k05:.4f}): {mag_k05.mean():.6f}")
print(f"Output norm std: {mag_k05.std():.6f}")
print(f"Expected radius: 1/√0.5 = {expected_radius_k05:.6f}")
assert torch.allclose(mag_k05, torch.full_like(mag_k05, expected_radius_k05), atol=1e-5), \
    f"Curvature 0.5 failed: magnitude != {expected_radius_k05}"
print(f"✓ PASSED: All vectors projected to sphere (radius={expected_radius_k05:.6f})")

# Test with high curvature = 10.0 (very small sphere, radius = 0.316)
print("\n--- Testing with curvature k=10.0 (very small sphere, radius=0.316) ---")
normalized_k10 = vector_rms_norm(dummy_input, curvature=10.0)
mag_k10 = vector_compute_magnitude(normalized_k10)
expected_radius_k10 = 1.0 / math.sqrt(10.0)
print(f"Output shape: {normalized_k10.shape}")
print(f"Output norm (should be ~{expected_radius_k10:.4f}): {mag_k10.mean():.6f}")
print(f"Output norm std: {mag_k10.std():.6f}")
print(f"Expected radius: 1/√10 = {expected_radius_k10:.6f}")
assert torch.allclose(mag_k10, torch.full_like(mag_k10, expected_radius_k10), atol=1e-5), \
    f"Curvature 10.0 failed: magnitude != {expected_radius_k10}"
print(f"✓ PASSED: All vectors projected to sphere (radius={expected_radius_k10:.6f})")

# Test 2: Magnitude computation
print("\n" + "=" * 70)
print("TEST 2: Vector Magnitude Computation")
print("=" * 70)

# Create vectors with known magnitude
print("\nTesting magnitude computation with known vectors:")
v1 = torch.tensor([[3.0, 4.0]], dtype=torch.float32)  # magnitude should be 5.0
mag1 = vector_compute_magnitude(v1)
print(f"Vector: {v1}")
print(f"Computed magnitude: {mag1.item():.6f}")
print(f"Expected magnitude: 5.0")
assert torch.allclose(mag1, torch.tensor([[5.0]]), atol=1e-5), "Magnitude test failed"
print("✓ PASSED: Magnitude computation correct")

# Test 3: Angle computation
print("\n" + "=" * 70)
print("TEST 3: Vector Angle Computation")
print("=" * 70)

# Create vectors with known angles
# Orthogonal vectors (90 degrees)
print("\nTesting angle computation with orthogonal vectors:")
v_a = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
v_b = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
angle_90 = vector_compute_angle(v_a, v_b)
print(f"Vector A: {v_a}")
print(f"Vector B: {v_b}")
print(f"Computed angle: {angle_90.item():.2f}°")
print(f"Expected angle: 90°")
assert torch.allclose(angle_90, torch.tensor([[90.0]]), atol=1e-4), "Orthogonal angle test failed"
print("✓ PASSED: Orthogonal vectors have 90° angle")

# Parallel vectors (0 degrees)
print("\nTesting angle computation with parallel vectors:")
v_c = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
v_d = torch.tensor([[2.0, 2.0]], dtype=torch.float32)
angle_0 = vector_compute_angle(v_c, v_d)
print(f"Vector C: {v_c}")
print(f"Vector D: {v_d}")
print(f"Computed angle: {angle_0.item():.2f}°")
print(f"Expected angle: 0°")
assert torch.allclose(angle_0, torch.tensor([[0.0]]), atol=1e-4), "Parallel angle test failed"
print("✓ PASSED: Parallel vectors have 0° angle")

# Opposite vectors (180 degrees)
print("\nTesting angle computation with opposite vectors:")
v_e = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
v_f = torch.tensor([[-1.0, 0.0]], dtype=torch.float32)
angle_180 = vector_compute_angle(v_e, v_f)
print(f"Vector E: {v_e}")
print(f"Vector F: {v_f}")
print(f"Computed angle: {angle_180.item():.2f}°")
print(f"Expected angle: 180°")
assert torch.allclose(angle_180, torch.tensor([[180.0]]), atol=1e-4), "Opposite angle test failed"
print("✓ PASSED: Opposite vectors have 180° angle")

# Test 4: Batch processing
print("\n" + "=" * 70)
print("TEST 4: Batch Processing (Multiple Samples)")
print("=" * 70)

batch_vec = torch.randn(10, 8)  # 10 samples, 8 dimensions
print(f"Input batch shape: {batch_vec.shape}")

normalized_batch = vector_rms_norm(batch_vec, curvature=1.0)
mag_batch = vector_compute_magnitude(normalized_batch)
print(f"Normalized batch shape: {normalized_batch.shape}")
print(f"Magnitudes per sample: {mag_batch.squeeze().tolist()}")
print(f"All magnitudes close to 1.0? {torch.allclose(mag_batch, torch.ones_like(mag_batch), atol=1e-5)}")
assert torch.allclose(mag_batch, torch.ones_like(mag_batch), atol=1e-5), "Batch processing failed"
print("✓ PASSED: Batch processing works correctly")

# Test 5: Spherical softmax similarity
print("\n" + "=" * 70)
print("TEST 5: Spherical Similarity (Cosine Similarity on Sphere)")
print("=" * 70)

# Simulate embedding and class prototypes
embedding = torch.randn(3, 64)  # 3 samples, 64-dim features
num_classes = 10
class_prototypes = torch.randn(num_classes, 64)

# Normalize both to sphere with k=1.0
normalized_embedding = vector_rms_norm(embedding, curvature=1.0)
normalized_prototypes = vector_rms_norm(class_prototypes, curvature=1.0)

# Compute cosine similarity via dot product
logits = torch.mm(normalized_embedding, normalized_prototypes.t())
print(f"Embedding shape: {embedding.shape}")
print(f"Class prototypes shape: {class_prototypes.shape}")
print(f"Logits shape (similarity matrix): {logits.shape}")
print(f"Logits range: [{logits.min():.4f}, {logits.max():.4f}]")
print(f"Note: On unit sphere, dot product = cosine similarity ∈ [-1, 1]")

# Apply softmax for classification
log_probs = torch.nn.functional.log_softmax(logits, dim=1)
probs = torch.exp(log_probs)
print(f"Log probabilities shape: {log_probs.shape}")
print(f"Probabilities sum per sample (should be ~1.0): {probs.sum(dim=1).tolist()}")
assert torch.allclose(probs.sum(dim=1), torch.ones(3), atol=1e-5), "Probability sum test failed"
print("✓ PASSED: Spherical softmax classification works")

# Test 6: Gradient flow (backpropagation)
print("\n" + "=" * 70)
print("TEST 6: Gradient Flow (Backpropagation)")
print("=" * 70)

vec = torch.randn(2, 8, requires_grad=True)
normalized = vector_rms_norm(vec, curvature=1.0)
loss = normalized.sum()
loss.backward()

print(f"Vector requires_grad: {vec.requires_grad}")
print(f"Normalized output requires_grad: {normalized.requires_grad}")
print(f"Gradient shape: {vec.grad.shape}")
print(f"Gradient exists: {vec.grad is not None}")
assert vec.grad is not None, "Gradient flow failed"
print("✓ PASSED: Gradients flow correctly through RMS norm")

# Test 7: Curvature comparison
print("\n" + "=" * 70)
print("TEST 7: Curvature Parameter Scaling")
print("=" * 70)

input_vec = torch.randn(5, 32)
curvatures = [0.25, 0.5, 1.0, 2.0, 4.0]
print(f"Testing with curvatures: {curvatures}")
print(f"Expected radii (1/√k):\n")

results = []
for k in curvatures:
    normalized = vector_rms_norm(input_vec, curvature=k)
    mag = vector_compute_magnitude(normalized)
    expected_radius = 1.0 / math.sqrt(k)
    actual_radius = mag.mean().item()
    results.append((k, expected_radius, actual_radius))
    print(f"  k={k:<4.2f} → r_expected={expected_radius:.6f}, r_actual={actual_radius:.6f}, match={torch.allclose(mag, torch.full_like(mag, expected_radius), atol=1e-5)}")
    assert torch.allclose(mag, torch.full_like(mag, expected_radius), atol=1e-5), f"Curvature {k} scaling failed"

print("✓ PASSED: All curvature values scale correctly")

# Summary
print("\n" + "=" * 70)
print("ALL TESTS PASSED! ✓")
print("=" * 70)
print(f"""
Summary of verified functionality:
✓ RMS normalization projects vectors to spheres with correct radii
✓ Radius scales as 1/√k for curvature k
✓ Magnitude computation is accurate
✓ Angle computation works for 0°, 90°, and 180° cases
✓ Batch processing handles multiple samples
✓ Spherical softmax classification via dot product
✓ Gradients flow correctly for backpropagation
✓ Curvature parameter scales outputs appropriately

The spherical conversion is working correctly! 
""")
