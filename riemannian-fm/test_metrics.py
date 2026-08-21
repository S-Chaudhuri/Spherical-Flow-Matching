import torch
from geomloss import SamplesLoss

from manifm.metrics import ManifoldMetricHandler


# ============================================================
# Configuration
# ============================================================

SEED = 34
DIM = 2
N = 1000
STD = 0.3

# Start with Poincare.
# Change to "sphere" for the second test.
MANIFOLD = "sphere"
CURVATURE = 1.0

BLURS = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]

# Number of samples used for pairwise distance diagnostics.
N_DIAG = 100


# ============================================================
# Handler
# ============================================================

def make_handler(manifold, dim, curvature):
    cfg = {
        "general": {
            "manifold": manifold,
            "dim": dim,
            "curvature": curvature,
        },
        "metrics_used": {},
        "metrics_param": {},
        "cross_curvature": False,
    }

    return ManifoldMetricHandler(cfg)


# ============================================================
# Generate valid samples
# ============================================================

def make_samples(handler, n, dim, std=0.3):
    """
    Generate samples around the canonical origin.

    Euclidean:
        Gaussian samples directly.

    Poincare / Sphere:
        Gaussian samples in the tangent space followed by expmap.
    """

    tangent = torch.randn(n, dim) * std

    if handler.m_type == "euclidean":
        return tangent

    origin = handler.get_origin(tangent)

    # At the spherical north pole, the first coordinate is normal
    # to the tangent space.
    if handler.m_type == "sphere":
        tangent[:, 0] = 0.0

    origin = origin.expand_as(tangent)

    samples = handler.manifold.expmap(
        origin,
        tangent,
    )

    if hasattr(handler.manifold, "projx"):
        samples = handler.manifold.projx(samples)

    return samples


# ============================================================
# Pairwise distance diagnostics
# ============================================================

def inspect_distances(handler, pred, target):
    print("\n" + "=" * 70)
    print("1. PAIRWISE DISTANCE DIAGNOSTICS")
    print("=" * 70)

    x = pred[:N_DIAG]
    y = target[:N_DIAG]

    x_exp = x.unsqueeze(1)
    y_exp = y.unsqueeze(0)

    # --------------------------------------------------------
    # Geodesic
    # --------------------------------------------------------

    geo = handler.manifold.dist(x_exp, y_exp)

    print("\nGeodesic distance:")
    print(f"  shape  : {tuple(geo.shape)}")
    print(f"  min    : {geo.min().item():.8f}")
    print(f"  median : {geo.median().item():.8f}")
    print(f"  mean   : {geo.mean().item():.8f}")
    print(f"  max    : {geo.max().item():.8f}")

    # --------------------------------------------------------
    # Tangent
    # --------------------------------------------------------

    v_x = handler.tangent_coordinates_scaled(x)
    v_y = handler.tangent_coordinates_scaled(y)

    tangent = torch.cdist(v_x, v_y)

    print("\nTangent-space distance:")
    print(f"  shape  : {tuple(tangent.shape)}")
    print(f"  min    : {tangent.min().item():.8f}")
    print(f"  median : {tangent.median().item():.8f}")
    print(f"  mean   : {tangent.mean().item():.8f}")
    print(f"  max    : {tangent.max().item():.8f}")

    # --------------------------------------------------------
    # Ratio
    # --------------------------------------------------------

    # ratio = geo / torch.clamp(tangent, min=1e-8)

    # print("\nGeodesic / tangent distance ratio:")
    # print(f"  median : {ratio.median().item():.8f}")
    # print(f"  mean   : {ratio.mean().item():.8f}")
    # print(f"  max    : {ratio.max().item():.8f}")


    diff = (geo - tangent).abs()

    print("max absolute difference:", diff.max())
    print("mean absolute difference:", diff.mean())

# ============================================================
# Distance symmetry
# ============================================================

def check_symmetry(handler, samples):
    print("\n" + "=" * 70)
    print("2. GEODESIC DISTANCE SYMMETRY")
    print("=" * 70)

    x = samples[:N_DIAG]

    x_exp = x.unsqueeze(1)
    y_exp = x.unsqueeze(0)

    d_xy = handler.manifold.dist(x_exp, y_exp)
    d_yx = handler.manifold.dist(y_exp, x_exp)

    error = (d_xy - d_yx.T).abs()

    print(f"\nmax |d(x,y) - d(y,x)| : {error.max().item():.12e}")
    print(f"mean absolute error    : {error.mean().item():.12e}")

    if error.max() < 1e-5:
        print("RESULT: PASS")
    else:
        print("RESULT: WARNING")


# ============================================================
# Sinkhorn blur sensitivity
# ============================================================

def test_sinkhorn_blur(handler, pred, target):
    print("\n" + "=" * 70)
    print("3. SINKHORN BLUR SENSITIVITY")
    print("=" * 70)

    print("\n   blur          geodesic Sinkhorn")
    print("   --------------------------------")

    for blur in BLURS:
        value = handler.calculate_sinkhorn_divergence(
            pred,
            target,
            p=1,
            blur=blur,
        )

        print(f"   {blur:7.3f}        {value.item():.10f}")


# ============================================================
# Geodesic vs tangent Sinkhorn
# ============================================================

def compare_sinkhorn(handler, pred, target):
    print("\n" + "=" * 70)
    print("4. GEODESIC VS TANGENT SINKHORN")
    print("=" * 70)

    geo = handler.calculate_sinkhorn_divergence(
        pred,
        target,
        p=1,
        blur=0.05,
    )

    tangent = handler.calculate_tangent_sinkhorn(
        pred,
        target,
    )

    print(f"\nGeodesic Sinkhorn : {geo.item():.10f}")
    print(f"Tangent Sinkhorn  : {tangent.item():.10f}")
    print(f"Ratio geo/tangent : {(geo / tangent).item():.10f}")


# ============================================================
# Geodesic vs tangent MMD
# ============================================================

def compare_mmd(handler, pred, target):
    print("\n" + "=" * 70)
    print("5. GEODESIC VS TANGENT MMD")
    print("=" * 70)

    geo = handler.calculate_mmd(
        pred,
        target,
    )

    tangent = handler.calculate_tangent_mmd(
        pred,
        target,
    )

    print(f"\nGeodesic MMD      : {geo.item():.10f}")
    print(f"Tangent MMD       : {tangent.item():.10f}")
    print(f"Ratio geo/tangent : {(geo / tangent).item():.10f}")


# ============================================================
# Explicit geodesic scaling test
# ============================================================

def test_scaled_geodesic_sinkhorn(handler, pred, target):
    print("\n" + "=" * 70)
    print("6. GEODESIC SINKHORN WITH EXPLICIT DISTANCE SCALING")
    print("=" * 70)

    print("\n   scale       Sinkhorn")
    print("   ----------------------")

    for scale in [0.25, 0.5, 1.0, 2.0]:

        def scaled_geodesic_cost(x, y):
            x_exp = x.unsqueeze(1)
            y_exp = y.unsqueeze(0)

            d = handler.manifold.dist(
                x_exp,
                y_exp,
            )

            return scale * d

        solver = SamplesLoss(
            loss="sinkhorn",
            p=1,
            blur=0.05,
            debias=True,
            cost=scaled_geodesic_cost,
            backend="tensorized",
        )

        value = solver(pred, target)

        print(
            f"   {scale:5.2f}       "
            f"{value.item():.10f}"
        )

def test_geomloss_custom_cost_equivalence(handler, pred, target):
    print("\n" + "=" * 70)
    print("7. GEOMLOSS CUSTOM COST EQUIVALENCE TEST")
    print("=" * 70)

    blur = 0.05

    # --------------------------------------------------------
    # Tangent coordinates
    # --------------------------------------------------------

    v_pred = handler.tangent_coordinates_scaled(pred)
    v_target = handler.tangent_coordinates_scaled(target)

    # --------------------------------------------------------
    # Explicit tangent distance matrix
    # --------------------------------------------------------

    tangent_cost = torch.cdist(
        v_pred,
        v_target,
        p=2,
    )

    print("\nTangent distance matrix:")
    print(f"  min    : {tangent_cost.min().item():.10f}")
    print(f"  median : {tangent_cost.median().item():.10f}")
    print(f"  mean   : {tangent_cost.mean().item():.10f}")
    print(f"  max    : {tangent_cost.max().item():.10f}")

    # --------------------------------------------------------
    # Geodesic distance matrix
    # --------------------------------------------------------

    geo_cost = handler.manifold.dist(
        pred.unsqueeze(1),
        target.unsqueeze(0),
    )

    print("\nGeodesic distance matrix:")
    print(f"  min    : {geo_cost.min().item():.10f}")
    print(f"  median : {geo_cost.median().item():.10f}")
    print(f"  mean   : {geo_cost.mean().item():.10f}")
    print(f"  max    : {geo_cost.max().item():.10f}")

    # --------------------------------------------------------
    # Direct comparison
    # --------------------------------------------------------

    diff = (geo_cost - tangent_cost).abs()

    print("\nGeodesic vs tangent matrix difference:")
    print(f"  max abs error  : {diff.max().item():.12e}")
    print(f"  mean abs error : {diff.mean().item():.12e}")

    # --------------------------------------------------------
    # Normal Euclidean GeomLoss
    # --------------------------------------------------------

    solver_euclidean = SamplesLoss(
        loss="sinkhorn",
        p=1,
        blur=blur,
        debias=True,
        backend="tensorized",
    )

    result_euclidean = solver_euclidean(
        v_pred,
        v_target,
    )

    print("\nNormal Euclidean SamplesLoss:")
    print(f"  result = {result_euclidean.item():.10f}")

    # --------------------------------------------------------
    # Custom cost that does EXACTLY the same Euclidean
    # distance calculation
    # --------------------------------------------------------

    def euclidean_custom_cost(x, y):
        return torch.cdist(x, y, p=2)

    solver_custom_euclidean = SamplesLoss(
        loss="sinkhorn",
        p=1,
        blur=blur,
        debias=True,
        cost=euclidean_custom_cost,
        backend="tensorized",
    )

    result_custom_euclidean = solver_custom_euclidean(
        v_pred,
        v_target,
    )

    print("\nCustom-cost Euclidean SamplesLoss:")
    print(f"  result = {result_custom_euclidean.item():.10f}")

    print("\nDifference:")
    print(
        f"  custom - normal = "
        f"{(result_custom_euclidean - result_euclidean).item():.12e}"
    )

    # --------------------------------------------------------
    # Custom geodesic cost
    # --------------------------------------------------------

    def geodesic_custom_cost(x, y):
        return handler.manifold.dist(
            x.unsqueeze(1),
            y.unsqueeze(0),
        )

    solver_custom_geodesic = SamplesLoss(
        loss="sinkhorn",
        p=1,
        blur=blur,
        debias=True,
        cost=geodesic_custom_cost,
        backend="tensorized",
    )

    result_custom_geodesic = solver_custom_geodesic(
        pred,
        target,
    )

    print("\nCustom geodesic SamplesLoss:")
    print(f"  result = {result_custom_geodesic.item():.10f}")


    # --------------------------------------------------------
    # Standard handler results
    # --------------------------------------------------------

    tangent_handler = handler.calculate_tangent_sinkhorn(
        pred,
        target,
        p=1,
        blur=blur,
    )

    geo_handler = handler.calculate_sinkhorn_divergence(
        pred,
        target,
        p=1,
        blur=blur,
    )

    print("\nHandler results:")
    print(f"  tangent handler = {tangent_handler.item():.10f}")
    print(f"  geodesic handler = {geo_handler.item():.10f}")

def test_debias_effect(handler, pred, target):
    print("\n" + "=" * 70)
    print("8. EFFECT OF SINKHORN DEBIASING")
    print("=" * 70)

    blur = 0.05

    # --------------------------------------------------------
    # Tangent coordinates
    # --------------------------------------------------------

    v_pred = handler.tangent_coordinates_scaled(pred)
    v_target = handler.tangent_coordinates_scaled(target)

    # --------------------------------------------------------
    # Euclidean cost
    # --------------------------------------------------------

    print("\nTANGENT / EUCLIDEAN")

    for debias in [False, True]:

        solver = SamplesLoss(
            loss="sinkhorn",
            p=1,
            blur=blur,
            debias=debias,
            backend="tensorized",
        )

        value = solver(v_pred, v_target)

        print(
            f"  debias={str(debias):5s} : "
            f"{value.item():.10f}"
        )

    # --------------------------------------------------------
    # Geodesic cost
    # --------------------------------------------------------

    print("\nSPHERE / GEODESIC")

    def geodesic_cost(x, y):
        return handler.manifold.dist(
            x.unsqueeze(1),
            y.unsqueeze(0),
        )

    for debias in [False, True]:

        solver = SamplesLoss(
            loss="sinkhorn",
            p=1,
            blur=blur,
            debias=debias,
            cost=geodesic_cost,
            backend="tensorized",
        )

        value = solver(pred, target)

        print(
            f"  debias={str(debias):5s} : "
            f"{value.item():.10f}"
        )

def test_precomputed_cost(handler, pred, target):
    print("\n" + "=" * 70)
    print("9. GEODESIC COST CALLBACK DIAGNOSTIC")
    print("=" * 70)

    blur = 0.05

    # Use a smaller subset.
    x = pred[:200]
    y = target[:200]

    # --------------------------------------------------------
    # Explicit geodesic matrix
    # --------------------------------------------------------

    geo = handler.manifold.dist(
        x.unsqueeze(1),
        y.unsqueeze(0),
    )

    print("\nExplicit geodesic matrix:")
    print(f"  shape  : {tuple(geo.shape)}")
    print(f"  min    : {geo.min().item():.10f}")
    print(f"  median : {geo.median().item():.10f}")
    print(f"  mean   : {geo.mean().item():.10f}")
    print(f"  max    : {geo.max().item():.10f}")

    # --------------------------------------------------------
    # Explicit tangent matrix
    # --------------------------------------------------------

    vx = handler.tangent_coordinates_scaled(x)
    vy = handler.tangent_coordinates_scaled(y)

    tangent = torch.cdist(vx, vy)

    diff = (geo - tangent).abs()

    print("\nGeodesic vs tangent:")
    print(f"  max abs diff  : {diff.max().item():.10f}")
    print(f"  mean abs diff : {diff.mean().item():.10f}")

    # --------------------------------------------------------
    # 1. Normal Euclidean GeomLoss
    # --------------------------------------------------------

    solver_euclidean = SamplesLoss(
        loss="sinkhorn",
        p=1,
        blur=blur,
        debias=True,
        backend="tensorized",
    )

    euclidean_result = solver_euclidean(
        vx,
        vy,
    )

    print("\nNormal Euclidean cost:")
    print(f"  Sinkhorn = {euclidean_result.item():.10f}")

    # --------------------------------------------------------
    # 2. Custom Euclidean cost
    # --------------------------------------------------------

    def euclidean_cost(a, b):
        return torch.cdist(a, b, p=2)

    solver_custom_euclidean = SamplesLoss(
        loss="sinkhorn",
        p=1,
        blur=blur,
        debias=True,
        cost=euclidean_cost,
        backend="tensorized",
    )

    custom_euclidean_result = solver_custom_euclidean(
        vx,
        vy,
    )

    print("\nCustom Euclidean cost:")
    print(f"  Sinkhorn = {custom_euclidean_result.item():.10f}")

    print(
        f"  difference = "
        f"{(custom_euclidean_result - euclidean_result).item():.12e}"
    )

    # --------------------------------------------------------
    # 3. Custom geodesic cost
    # --------------------------------------------------------

    def geodesic_cost(a, b):
        return handler.manifold.dist(
            a.unsqueeze(1),
            b.unsqueeze(0),
        )

    solver_geodesic = SamplesLoss(
        loss="sinkhorn",
        p=1,
        blur=blur,
        debias=True,
        cost=geodesic_cost,
        backend="tensorized",
    )

    geodesic_result = solver_geodesic(
        x,
        y,
    )

    print("\nCustom geodesic cost:")
    print(f"  Sinkhorn = {geodesic_result.item():.10f}")

    # --------------------------------------------------------
    # 4. Perturbed Euclidean cost
    #
    # This is useful because your sphere geodesic distance
    # differs from tangent distance only slightly.
    # --------------------------------------------------------

    for alpha in [0.0, 0.25, 0.5, 1.0]:

        def perturbed_cost(a, b, alpha=alpha):
            euclidean = torch.cdist(a, b, p=2)

            # We need to interpret a,b as sphere points here.
            geodesic = handler.manifold.dist(
                a.unsqueeze(1),
                b.unsqueeze(0),
            )

            return (1.0 - alpha) * euclidean + alpha * geodesic

        solver = SamplesLoss(
            loss="sinkhorn",
            p=1,
            blur=blur,
            debias=True,
            cost=perturbed_cost,
            backend="tensorized",
        )

        result = solver(x, y)

        print(
            f"\nInterpolated cost alpha={alpha:.2f}:"
        )
        print(
            f"  Sinkhorn = {result.item():.10f}"
        )
def test_sinkhorn_components(handler, pred, target):
    print("\n" + "=" * 70)
    print("10. SINKHORN COMPONENT DIAGNOSTIC")
    print("=" * 70)

    blur = 0.05

    # Use the same number of samples for all terms.
    x = pred[:200]
    y = target[:200]

    def euclidean_cost(a, b):
        return torch.cdist(a, b, p=2)

    def geodesic_cost(a, b):
        return handler.manifold.dist(
            a.unsqueeze(1),
            b.unsqueeze(0),
        )

    def run_components(name, cost_fn, a, b):

        print(f"\n{name}")

        # ----------------------------------------------------
        # Cross
        # ----------------------------------------------------

        cross_solver = SamplesLoss(
            loss="sinkhorn",
            p=1,
            blur=blur,
            debias=False,
            cost=cost_fn,
            backend="tensorized",
        )

        cross = cross_solver(a, b)

        # ----------------------------------------------------
        # Self A
        # ----------------------------------------------------

        self_a = cross_solver(a, a)

        # ----------------------------------------------------
        # Self B
        # ----------------------------------------------------

        self_b = cross_solver(b, b)

        # ----------------------------------------------------
        # Debiased combination
        # ----------------------------------------------------

        divergence = (
            cross
            - 0.5 * self_a
            - 0.5 * self_b
        )

        print(f"  OT(x, y)       = {cross.item():.10f}")
        print(f"  OT(x, x)       = {self_a.item():.10f}")
        print(f"  OT(y, y)       = {self_b.item():.10f}")
        print(f"  reconstructed  = {divergence.item():.10f}")

        # Compare with GeomLoss's debiased result.
        debiased_solver = SamplesLoss(
            loss="sinkhorn",
            p=1,
            blur=blur,
            debias=True,
            cost=cost_fn,
            backend="tensorized",
        )

        direct = debiased_solver(a, b)

        print(f"  GeomLoss       = {direct.item():.10f}")
        print(
            f"  difference     = "
            f"{(direct - divergence).item():.12e}"
        )

        return cross, self_a, self_b, divergence

    # --------------------------------------------------------
    # Euclidean / chord distance
    # --------------------------------------------------------

    run_components(
        "EUCLIDEAN / CHORD",
        euclidean_cost,
        x,
        y,
    )

    # --------------------------------------------------------
    # Geodesic
    # --------------------------------------------------------

    run_components(
        "SPHERICAL GEODESIC",
        geodesic_cost,
        x,
        y,
    )

    print("\nIDENTICAL SET TEST")

    solver = SamplesLoss(
        loss="sinkhorn",
        p=1,
        blur=blur,
        debias=True,
        cost=geodesic_cost,
        backend="tensorized",
    )

    same = solver(x, x)

    print(
        f"Geodesic Sinkhorn(x, x) = "
        f"{same.item():.12f}"
    )

# ============================================================
# Main
# ============================================================

torch.manual_seed(SEED)

print("=" * 70)
print("SINKHORN / GEODESIC DIAGNOSTIC")
print("=" * 70)

print(f"\nManifold  : {MANIFOLD}")
print(f"Curvature : {CURVATURE}")
print(f"Dimension : {DIM}")
print(f"N         : {N}")
print(f"STD       : {STD}")
print(f"Seed      : {SEED}")

handler = make_handler(
    manifold=MANIFOLD,
    dim=DIM,
    curvature=CURVATURE,
)


# ============================================================
# Same distribution, independent samples
# ============================================================

target = make_samples(
    handler,
    n=N,
    dim=DIM,
    std=STD,
)

pred = make_samples(
    handler,
    n=N,
    dim=DIM,
    std=STD,
)


# ============================================================
# Diagnostics
# ============================================================

inspect_distances(
    handler,
    pred,
    target,
)

check_symmetry(
    handler,
    pred,
)

test_sinkhorn_blur(
    handler,
    pred,
    target,
)

compare_sinkhorn(
    handler,
    pred,
    target,
)

compare_mmd(
    handler,
    pred,
    target,
)

test_scaled_geodesic_sinkhorn(
    handler,
    pred,
    target,
)

test_geomloss_custom_cost_equivalence(
    handler,
    pred,
    target,
)

test_debias_effect(
    handler,
    pred,
    target,
)

test_precomputed_cost(
    handler,
    pred,
    target,
)

test_sinkhorn_components(
    handler,
    pred,
    target,
)

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)