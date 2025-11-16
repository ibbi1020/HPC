# KLT Feature Tracking Algorithm - Simple Explanation

## What is KLT?

**KLT** stands for **Kanade-Lucas-Tomasi** (named after the three researchers who developed it). It's an algorithm that can **follow interesting points** in a video or series of images. Think of it like this: if you're watching a soccer game, KLT can follow specific players or the ball as they move around the field, frame by frame.

## Why Do We Need This?

Imagine you're trying to:
- Track a person walking through a crowd in security footage
- Follow your eye movements for a VR headset
- Stitch multiple photos together to create a panorama
- Help a robot understand how it's moving through a room

All of these need to identify and track the same points across multiple images. That's what KLT does!

## The Big Picture: Two Main Steps

KLT has two main jobs:

### 1. **Finding Good Features** (Feature Selection)
First, it looks at an image and finds "interesting" spots to track. These are usually **corners** - places where there's a lot of detail and texture.

**Why corners?** Because they're easy to recognize! Think about it:
- A blank wall? Hard to track - it all looks the same
- A single edge? Still tricky - which part of the edge are you following?
- A corner (like the corner of a window)? Perfect! It's unique and easy to spot again

### 2. **Tracking Those Features** (Feature Tracking)
Once it knows what to look for, KLT follows those points from one image to the next, calculating where each point moved to.

---

## How Does Feature Selection Work?

### The Process (Simplified)

1. **Convert to Numbers**: The image is converted into a grid of numbers (grayscale values), where each pixel has a brightness value from 0 (black) to 255 (white).

2. **Calculate Gradients**: The algorithm calculates how quickly brightness changes in both horizontal (x) and vertical (y) directions at each pixel. This is like finding the "slopes" in the image.
   - A flat area → small gradients (boring!)
   - An edge → large gradient in one direction
   - A corner → large gradients in BOTH directions (exciting!)

3. **Create a Small Window**: For each potential feature point, the algorithm looks at a small window around it (like a 7×7 pixel square).

4. **Calculate "Cornerness"**: Using the gradients in that window, it builds a mathematical matrix and calculates something called the "minimum eigenvalue." 
   - **Don't panic about the math!** Just know that a high eigenvalue = a strong corner = a good feature to track
   - Low eigenvalue = flat or edge-like = bad for tracking

5. **Pick the Best**: The algorithm sorts all potential points by their eigenvalue scores and picks the best ones (usually the top 100-300 points).

6. **Spread Them Out**: To avoid having all features bunched together in one spot, it enforces a minimum distance between features. If two good features are too close, it keeps the better one.

### Functions Involved in Feature Selection

**File: `selectGoodFeatures.c`**

The main entry point is **`KLTSelectGoodFeatures()`**, which orchestrates the entire selection process. Here's how the functions work together:

#### 1. Image Preparation
- **`_KLTToFloatImage()`** (in `convolve.c`): Converts the input image from bytes (0-255) to floating-point numbers for precise calculations
- **`_KLTComputeSmoothedImage()`** (in `convolve.c`): Smooths the image to reduce noise
  - Uses **`_convolveSeparate()`** which applies a Gaussian blur
  - Calls **`_convolveImageHoriz()`** and **`_convolveImageVert()`** for efficient 2D smoothing

#### 2. Gradient Computation
- **`_KLTComputeGradients()`** (in `convolve.c`): Calculates how brightness changes in x and y directions
  - Calls **`_computeKernels()`** to create Gaussian and Gaussian-derivative filters
  - Uses **`_convolveSeparate()`** twice:
    - Once with `[gaussderiv, gauss]` kernels → horizontal gradient (gradx)
    - Once with `[gauss, gaussderiv]` kernels → vertical gradient (grady)

#### 3. Feature Quality Evaluation
**Main function: `_KLTSelectGoodFeatures()`** does the heavy lifting:

- For each potential pixel location:
  - **Nested loops** iterate through all pixels (skipping borders and optionally some pixels for speed)
  - For each pixel, a window (e.g., 7×7) is examined
  - **`_minEigenvalue()`**: Calculates the "cornerness" score
    - Takes the gradient values in the window: `gxx` (sum of gradx²), `gxy` (sum of gradx*grady), `gyy` (sum of grady²)
    - Computes: `(gxx + gyy - sqrt((gxx - gyy)² + 4*gxy²)) / 2`
    - This is the minimum eigenvalue of the structure tensor matrix
  - Stores `[x, y, eigenvalue]` in a **pointlist**

#### 4. Sorting and Selection
- **`_sortPointList()`**: Sorts all potential features by eigenvalue (best first)
  - Uses **`_quicksort()`**: Custom quicksort implementation optimized for our 3-element entries
  - Alternative: can use standard `qsort()` with **`_comparePoints()`** comparison function

#### 5. Enforcing Spatial Distribution
- **`_enforceMinimumDistance()`**: Ensures features are well-distributed
  - Creates a boolean **featuremap** array (same size as image)
  - Processes features in order of quality (best to worst):
    - Checks if location is already "occupied" in the featuremap
    - If free and eigenvalue is good enough, accepts the feature
    - Calls **`_fillFeaturemap()`** to mark surrounding area as occupied (radius = mindist)
    - Rejects features too close to already-selected ones
  - Fills remaining feature slots with `-1` (not found)

#### Function Call Flow for Feature Selection:
```
KLTSelectGoodFeatures()
└── _KLTSelectGoodFeatures()
    ├── _KLTToFloatImage()
    ├── _KLTComputeSmoothedImage()
    │   └── _convolveSeparate()
    │       ├── _convolveImageHoriz()
    │       └── _convolveImageVert()
    ├── _KLTComputeGradients()
    │   ├── _computeKernels()
    │   └── _convolveSeparate() [2x for gradx and grady]
    ├── _minEigenvalue() [called thousands of times]
    ├── _sortPointList()
    │   └── _quicksort()
    └── _enforceMinimumDistance()
        └── _fillFeaturemap()
```

### What Makes a "Good" Feature?

A good feature has:
- ✅ High contrast (clear brightness changes)
- ✅ Unique patterns (not repetitive)
- ✅ Strong gradients in multiple directions (cornerness)
- ✅ Sufficient distance from other features

---

## How Does Feature Tracking Work?

Once we have good features in Frame 1, we need to find where they moved to in Frame 2.

### The Core Idea: Template Matching

Imagine you have a small photo of someone's face. To find them in a crowd photo, you'd slide your small photo around until it matches. KLT does something similar, but smarter!

### The Process (Simplified)

1. **Extract a Window**: Around each feature in Frame 1, extract a small window (like 7×7 pixels). This is your "template" - what you're looking for.

2. **Make an Initial Guess**: Start by guessing the feature is in approximately the same location in Frame 2 (people don't teleport between frames!).

3. **Refine Using Math** (The Newton-Raphson Method):
   - Compare the window from Frame 1 with the window at your current guess in Frame 2
   - Calculate how different they are (the "error" or "residue")
   - Calculate the image gradients to figure out which direction would reduce the error
   - Move the window in that direction
   - Repeat until the window matches well or you've tried enough times

4. **Check If It Worked**: The algorithm checks several things:
   - Is the feature still inside the image? (Out of bounds = lost)
   - Is the match good enough? (High residue = lost)
   - Did the math work? (Singular matrix = lost)

### Functions Involved in Feature Tracking

**File: `trackFeatures.c`**

The main entry point is **`KLTTrackFeatures()`**, which manages tracking all features across two images. Here's the detailed breakdown:

#### 1. Image Pyramid Creation

Before tracking begins, both images are preprocessed into multi-resolution pyramids:

- **`_KLTToFloatImage()`**: Converts input image to float
- **`_KLTComputeSmoothedImage()`**: Smooths the image
- **`_KLTCreatePyramid()`** (in `pyramid.c`): Allocates memory for pyramid structure
- **`_KLTComputePyramid()`** (in `pyramid.c`): Builds the pyramid
  - Level 0 = full resolution
  - Level 1 = subsampled (e.g., 1/2 or 1/4 size)
  - Level 2 = further subsampled
  - Each level is smoothed with **`_KLTComputeSmoothedImage()`** before subsampling
- **`_KLTComputeGradients()`**: Computes gradients for each pyramid level

**Why build pyramids for both images?**
- Pyramid 1 (previous frame) and Pyramid 2 (current frame) allow coarse-to-fine tracking
- Sequential mode optimization: If tracking consecutive frames, Pyramid 1 is cached from the previous iteration (stored in `tc->pyramid_last`)

#### 2. Per-Feature Tracking Loop

For each feature in the feature list, **`KLTTrackFeatures()`** calls the tracking logic:

**Main tracking function: `_trackFeature()`**

This function tracks a single feature through one pyramid level. Here's what happens:

##### A. Window Extraction and Comparison
- **`_computeIntensityDifference()`**: Compares windows between two images
  - For each pixel in the window (e.g., 7×7 = 49 pixels):
    - Calls **`_interpolate()`** to get the pixel value at position (x₁+i, y₁+j) in Frame 1
    - Calls **`_interpolate()`** to get the pixel value at position (x₂+i, y₂+j) in Frame 2
    - Computes difference: `imgdiff[i] = frame1_pixel - frame2_pixel`
  
- **`_interpolate()`**: Critical helper function
  - Gets pixel values at **sub-pixel positions** (e.g., x=10.5, y=20.3)
  - Uses **bilinear interpolation**: 
    - Finds the 4 surrounding pixels
    - Weights them based on distance: `(1-ax)*(1-ay)*topleft + ax*(1-ay)*topright + ...`
  - Allows smooth, precise tracking (not limited to integer pixel positions)

##### B. Gradient Calculation
- **`_computeGradientSum()`**: Computes the sum of gradients from both frames
  - For each pixel in the window:
    - Calls **`_interpolate()`** four times:
      - gradx₁ at (x₁+i, y₁+j)
      - gradx₂ at (x₂+i, y₂+j)
      - grady₁ at (x₁+i, y₁+j)
      - grady₂ at (x₂+i, y₂+j)
    - Sums them: `gradx = gradx₁ + gradx₂`, `grady = grady₁ + grady₂`

##### C. Building the Linear System
- **`_compute2by2GradientMatrix()`**: Builds the 2×2 matrix from gradients
  - Computes: 
    - `gxx = Σ(gradx²)` - sum of squared horizontal gradients
    - `gxy = Σ(gradx * grady)` - sum of cross-products
    - `gyy = Σ(grady²)` - sum of squared vertical gradients
  - This forms the "Z matrix" or "structure tensor"

- **`_compute2by1ErrorVector()`**: Builds the error vector
  - Computes:
    - `ex = Σ(imgdiff * gradx)` - how much error in x direction
    - `ey = Σ(imgdiff * grady)` - how much error in y direction
  - Multiplies by `step_factor` (usually 1.0 or 2.0)

##### D. Solving for Motion
- **`_solveEquation()`**: Solves the 2×2 linear system
  - Solves: `[gxx gxy; gxy gyy] * [dx; dy] = [ex; ey]`
  - Uses Cramer's rule: 
    - `det = gxx*gyy - gxy*gxy`
    - `dx = (gyy*ex - gxy*ey) / det`
    - `dy = (gxx*ey - gxy*ex) / det`
  - Returns `KLT_SMALL_DET` if determinant too small (unstable math)

##### E. Iteration Control
- **Main tracking loop** in `_trackFeature()`:
  - Repeats steps A-D until:
    - Displacement `(dx, dy)` is smaller than threshold (converged!)
    - Maximum iterations reached (give up)
    - Feature goes out of bounds (lost!)
    - Determinant becomes too small (lost!)

##### F. Final Validation
After iteration completes:
- **Boundary check**: Uses **`_outOfBounds()`** to verify feature is still in image
- **Residue check**: Calls `_computeIntensityDifference()` one final time
  - Uses **`_sumAbsFloatWindow()`** to calculate total difference
  - If `residue > max_residue`, marks as `KLT_LARGE_RESIDUE` (bad match)

#### 3. Pyramid Level Iteration

**`KLTTrackFeatures()`** orchestrates multi-resolution tracking:

```c
for each feature:
    for pyramid_level from coarsest to finest:
        scale coordinates to current level
        call _trackFeature() at this level
        if tracking failed: break (mark as lost)
        scale coordinates to next finer level
```

- Starts at coarsest level (small image, big pixels)
- Refines at each finer level
- If tracking fails at any level, feature is marked as lost

#### 4. Feature Status Management

After tracking each feature, **`KLTTrackFeatures()`** updates the feature list:
- Sets `feature->x` and `feature->y` to new position
- Sets `feature->val` to one of:
  - `KLT_TRACKED` (success!)
  - `KLT_OOB` (out of bounds)
  - `KLT_SMALL_DET` (math failed)
  - `KLT_LARGE_RESIDUE` (poor match)
  - `KLT_MAX_ITERATIONS` (didn't converge)

#### Function Call Flow for Feature Tracking:
```
KLTTrackFeatures()
├── Image Pyramid Setup (for both images)
│   ├── _KLTToFloatImage()
│   ├── _KLTComputeSmoothedImage()
│   ├── _KLTCreatePyramid()
│   ├── _KLTComputePyramid()
│   │   └── _KLTComputeSmoothedImage() [per level]
│   └── _KLTComputeGradients() [per level]
│
└── For each feature in featurelist:
    └── For each pyramid level (coarse to fine):
        └── _trackFeature()
            ├── Iterative refinement loop:
            │   ├── Boundary check: _outOfBounds()
            │   ├── _computeIntensityDifference()
            │   │   └── _interpolate() [called ~100 times per iteration]
            │   ├── _computeGradientSum()
            │   │   └── _interpolate() [called ~200 times per iteration]
            │   ├── _compute2by2GradientMatrix()
            │   ├── _compute2by1ErrorVector()
            │   └── _solveEquation()
            │
            └── Final validation:
                ├── _outOfBounds()
                ├── _computeIntensityDifference()
                └── _sumAbsFloatWindow()
```

#### Special Optimizations in the Code

**Sequential Mode**: The tracking context has a `sequentialMode` flag:
- When enabled, stores the previous frame's pyramid (`tc->pyramid_last`)
- On next frame, reuses this pyramid instead of rebuilding it
- Saves computation: one pyramid build per frame instead of two

**Memory Management**: 
- **`_allocateFloatWindow()`**: Allocates temporary windows for calculations
- Windows are freed after each feature is tracked to avoid memory leaks

**Lighting Invariance** (optional):
- **`_computeIntensityDifferenceLightingInsensitive()`**: Alternative to standard difference
  - Normalizes for brightness changes (gain and bias)
  - Useful when lighting changes between frames
- **`_computeGradientSumLightingInsensitive()`**: Gradient version
  - Adjusts gradients by gain factor `alpha`

### The Pyramid Trick

Here's a clever optimization: instead of searching pixel-by-pixel across the whole image, KLT uses an **image pyramid**:

```
Level 2:  [tiny blurry image]    ← Start here (coarse search)
              ↓
Level 1:  [medium image]         ← Refine here
              ↓
Level 0:  [full resolution]      ← Final precise tracking
```

Think of it like finding a friend in a stadium:
1. First, scan from far away to find the right section (coarse)
2. Move closer to find the right row (medium)
3. Get close to identify the exact seat (fine)

This is called **coarse-to-fine tracking**, and it's much faster than searching at full resolution!

---

## The Math Behind Tracking (Slightly More Detail)

Don't worry if this seems complex - it's okay to just understand the concept!

### The Basic Equation

KLT tries to solve this problem: "Where did pixel (x₁, y₁) from Frame 1 move to in Frame 2?"

It assumes that:
- **Brightness Constancy**: The brightness of a tracked point doesn't change much between frames
- **Small Motion**: The point doesn't move too far
- **Spatial Coherence**: Nearby points move together

The algorithm solves this equation:
```
I₁(x, y) ≈ I₂(x + dx, y + dy)
```

Where:
- `I₁` = intensity (brightness) in Frame 1
- `I₂` = intensity in Frame 2
- `dx, dy` = how far the point moved

### Iterative Refinement

Since the equation is complex, KLT solves it iteratively:
1. Start with a guess for (dx, dy)
2. Calculate how wrong the guess is
3. Use calculus (derivatives/gradients) to figure out how to improve
4. Update the guess
5. Repeat until it converges (or give up after max iterations)

---

## Main Components of the Code

### 1. **Data Structures**

- **`KLT_TrackingContext`**: Stores all the settings (window size, thresholds, pyramid levels, etc.)
- **`KLT_FeatureList`**: A list of features being tracked, each with:
  - `x, y`: Current position
  - `val`: Status (tracked, lost, etc.)
- **`KLT_Pyramid`**: Multi-resolution image representation for fast tracking

### 2. **Key Functions and Their Roles**

#### In `klt.c` - Main API and Setup:
- **`KLTCreateTrackingContext()`**: Initializes tracking parameters
  - Sets default values for all parameters (window size, thresholds, etc.)
  - Calls **`KLTChangeTCPyramid()`** to determine pyramid levels based on search range
  - Calls **`KLTUpdateTCBorder()`** to compute border regions
- **`KLTCreateFeatureList()`**: Allocates memory for feature list
- **`KLTFreeTrackingContext()`**, **`KLTFreeFeatureList()`**: Cleanup functions
- **`KLTCountRemainingFeatures()`**: Counts how many features are still tracked
- **`KLTChangeTCPyramid()`**: Calculates optimal pyramid configuration
  - Determines `nPyramidLevels` (usually 2-3)
  - Sets `subsampling` factor (2, 4, or 8)
  - Based on `search_range` and `window_size`

#### In `selectGoodFeatures.c` - Feature Detection:
- **`KLTSelectGoodFeatures()`**: Main entry point for feature selection
  - User-facing wrapper that calls internal function
  - Prints status messages if verbose mode enabled
- **`_KLTSelectGoodFeatures()`**: Core feature selection logic
  - Coordinates all steps: smoothing, gradients, evaluation, sorting, spacing
- **`_minEigenvalue()`**: Calculates "cornerness" score
  - Input: `gxx, gxy, gyy` (gradient matrix elements)
  - Output: Minimum eigenvalue (feature quality measure)
  - Formula: `(gxx + gyy - sqrt((gxx - gyy)² + 4*gxy²)) / 2`
- **`_quicksort()`**: Custom sorting for feature candidates
  - Sorts by eigenvalue in descending order
  - Optimized for 3-element entries `[x, y, value]`
- **`_enforceMinimumDistance()`**: Spatial distribution of features
  - Ensures features are at least `mindist` pixels apart
  - Uses featuremap to track occupied regions
- **`_fillFeaturemap()`**: Marks area around a feature as occupied
- **`KLTReplaceLostFeatures()`**: Finds new features to replace lost ones
  - Calls `_KLTSelectGoodFeatures()` in `REPLACING_SOME` mode
  - Only fills slots of lost features

#### In `trackFeatures.c` - Feature Tracking:
- **`KLTTrackFeatures()`**: Main entry point for tracking features
  - Manages multi-resolution tracking for all features
  - Handles pyramid creation for both frames
  - Loops through all features and pyramid levels
- **`_trackFeature()`**: Tracks a single feature at one pyramid level
  - Implements Newton-Raphson iterative refinement
  - Returns tracking status code
- **`_interpolate()`**: Bilinear interpolation at sub-pixel positions
  - **Most frequently called function** (thousands of times per frame!)
  - Computes weighted average of 4 neighboring pixels
  - Critical for accurate sub-pixel tracking
- **`_computeIntensityDifference()`**: Compares windows between frames
  - Calls `_interpolate()` for each pixel in both windows
  - Computes difference: `frame1 - frame2`
- **`_computeGradientSum()`**: Combines gradients from both frames
  - Calls `_interpolate()` four times per window pixel
  - Adds gradients: `gradx1 + gradx2`, `grady1 + grady2`
- **`_compute2by2GradientMatrix()`**: Builds structure tensor
  - Computes `gxx, gxy, gyy` from gradient windows
- **`_compute2by1ErrorVector()`**: Builds error vector
  - Computes `ex, ey` from intensity differences and gradients
- **`_solveEquation()`**: Solves 2×2 linear system
  - Calculates displacement `(dx, dy)` using Cramer's rule
- **`_outOfBounds()`**: Checks if feature is inside image boundaries
- **`_sumAbsFloatWindow()`**: Computes total residue (match quality)
- **`_allocateFloatWindow()`**: Memory allocation for temporary windows

**Lighting-Insensitive Variants** (optional features):
- **`_computeIntensityDifferenceLightingInsensitive()`**: Normalized comparison
- **`_computeGradientSumLightingInsensitive()`**: Normalized gradients

**Affine Tracking Functions** (advanced feature):
- **`_am_trackFeatureAffine()`**: Tracks with affine transformation
  - Handles rotation, scaling, shearing in addition to translation
  - Uses 6-parameter affine model instead of 2-parameter translation
- Multiple helper functions with `_am_` prefix for affine calculations

#### In `pyramid.c` - Multi-Resolution Processing:
- **`_KLTCreatePyramid()`**: Allocates pyramid structure
  - Creates array of images at different resolutions
  - Stores metadata (ncols, nrows per level)
- **`_KLTComputePyramid()`**: Fills pyramid with image data
  - Level 0: Copy of original image
  - Level 1+: Smoothed and subsampled versions
  - Calls `_KLTComputeSmoothedImage()` before each subsampling
  - Subsampling: Takes every Nth pixel (e.g., every 2nd or 4th)
- **`_KLTFreePyramid()`**: Deallocates pyramid memory

#### In `convolve.c` - Image Processing:
- **`_KLTToFloatImage()`**: Converts unsigned char image to float
  - Input: 0-255 byte values
  - Output: Float array for precise calculations
- **`_KLTComputeGradients()`**: Calculates image derivatives
  - Produces gradx (horizontal gradient) and grady (vertical gradient)
  - Uses separable Gaussian derivative filters
- **`_KLTComputeSmoothedImage()`**: Gaussian blur
  - Reduces noise before processing
  - Uses separable Gaussian filter
- **`_convolveSeparate()`**: 2D convolution via 1D passes
  - Applies horizontal kernel, then vertical kernel
  - More efficient than direct 2D convolution
- **`_convolveImageHoriz()`**: 1D horizontal convolution
  - Slides kernel across each row
- **`_convolveImageVert()`**: 1D vertical convolution
  - Slides kernel down each column
- **`_computeKernels()`**: Generates Gaussian and derivative kernels
  - Creates filter weights based on sigma
  - Normalizes kernels
- **`_KLTGetKernelWidths()`**: Returns kernel sizes for given sigma

#### In `klt_util.c` - Utility Functions:
- **`_KLTCreateFloatImage()`**: Allocates float image structure
- **`_KLTFreeFloatImage()`**: Deallocates float image
- **`_KLTWriteFloatImageToPGM()`**: Saves image to file (debugging)

#### In `storeFeatures.c` and `writeFeatures.c` - I/O:
- **`KLTWriteFeatureList()`**: Saves features to text file
- **`KLTWriteFeatureListToPPM()`**: Visualizes features on image
- **`KLTStoreFeatureList()`**: Stores features in feature table
- **`KLTWriteFeatureTable()`**: Saves tracking results across frames

### 3. **How Functions Work Together - Complete Example**

Let's trace what happens when you call the main functions:

#### **Initialization (One Time)**:
```c
tc = KLTCreateTrackingContext();
```
- Allocates context structure
- Sets default parameters
- Computes pyramid configuration
- Updates border sizes

```c
fl = KLTCreateFeatureList(nFeatures);
```
- Allocates array of feature structures
- Initializes feature positions to default values

#### **Frame 0 - Initial Feature Selection**:
```c
KLTSelectGoodFeatures(tc, img1, ncols, nrows, fl);
```

**Flow:**
1. **`KLTSelectGoodFeatures()`** → **`_KLTSelectGoodFeatures()`**
2. Convert and smooth: **`_KLTToFloatImage()`** → **`_KLTComputeSmoothedImage()`**
3. Compute gradients: **`_KLTComputeGradients()`**
   - → **`_convolveSeparate()`** (for gradx)
     - → **`_convolveImageHoriz()`** with Gaussian-derivative
     - → **`_convolveImageVert()`** with Gaussian
   - → **`_convolveSeparate()`** (for grady)
     - → **`_convolveImageHoriz()`** with Gaussian
     - → **`_convolveImageVert()`** with Gaussian-derivative
4. Evaluate all pixels:
   - For each pixel in valid region:
     - For each pixel in window around it:
       - Accumulate `gxx, gxy, gyy` from gradients
     - **`_minEigenvalue(gxx, gxy, gyy)`** → eigenvalue score
     - Store in pointlist
5. Sort: **`_sortPointList()`** → **`_quicksort()`**
6. Select: **`_enforceMinimumDistance()`**
   - For each point in sorted order:
     - Check featuremap
     - If free, accept and **`_fillFeaturemap()`**
7. Return feature list with selected features

#### **Frame 1+ - Tracking Features**:
```c
KLTTrackFeatures(tc, img1, img2, ncols, nrows, fl);
```

**Flow:**
1. **`KLTTrackFeatures()`** starts
2. Build pyramids for Frame 1:
   - **`_KLTCreatePyramid()`** allocates structure
   - **`_KLTToFloatImage()`** → **`_KLTComputeSmoothedImage()`**
   - **`_KLTComputePyramid()`** fills pyramid levels
   - **`_KLTComputeGradients()`** for each level
3. Build pyramids for Frame 2 (same process)
4. **For each feature** in feature list:
   - Scale coordinates to coarsest level
   - **For each pyramid level** (coarse to fine):
     - Call **`_trackFeature()`**:
       - **Iterate** up to max_iterations:
         - Check bounds: **`_outOfBounds()`**
         - **`_computeIntensityDifference()`**:
           - For each window pixel: **`_interpolate()`** twice
         - **`_computeGradientSum()`**:
           - For each window pixel: **`_interpolate()`** four times
         - **`_compute2by2GradientMatrix()`** from gradient windows
         - **`_compute2by1ErrorVector()`** from intensity difference
         - **`_solveEquation()`** → compute `(dx, dy)`
         - Update position: `x += dx, y += dy`
         - Check convergence: `if (|dx| < th && |dy| < th) break;`
       - Validate result:
         - **`_outOfBounds()`** check
         - **`_computeIntensityDifference()`** final check
         - **`_sumAbsFloatWindow()`** → residue
     - If tracking failed at any level: mark feature as lost, break
     - Scale to next finer level
   - Update feature: `feature->x`, `feature->y`, `feature->val`
5. Clean up pyramids (or cache for sequential mode)

### 4. **Function Interaction Diagram**

```
User Code
    |
    ├─→ KLTCreateTrackingContext()
    │       └─→ KLTChangeTCPyramid()
    │       └─→ KLTUpdateTCBorder()
    |
    ├─→ KLTSelectGoodFeatures()
    │       └─→ _KLTSelectGoodFeatures()
    │               ├─→ _KLTToFloatImage()
    │               ├─→ _KLTComputeSmoothedImage()
    │               │       └─→ _convolveSeparate()
    │               ├─→ _KLTComputeGradients()
    │               │       └─→ _convolveSeparate() [2x]
    │               ├─→ _minEigenvalue() [many times]
    │               ├─→ _sortPointList()
    │               └─→ _enforceMinimumDistance()
    |
    └─→ KLTTrackFeatures()
            ├─→ _KLTCreatePyramid()
            ├─→ _KLTComputePyramid()
            │       └─→ _KLTComputeSmoothedImage() [per level]
            ├─→ _KLTComputeGradients() [per level]
            └─→ [For each feature, each level]
                    └─→ _trackFeature()
                            ├─→ [Iterate]
                            │   ├─→ _computeIntensityDifference()
                            │   │       └─→ _interpolate() [many]
                            │   ├─→ _computeGradientSum()
                            │   │       └─→ _interpolate() [many]
                            │   ├─→ _compute2by2GradientMatrix()
                            │   ├─→ _compute2by1ErrorVector()
                            │   └─→ _solveEquation()
                            └─→ [Validate]
                                └─→ _sumAbsFloatWindow()
```

### 5. **Most Critical Functions by CPU Time**

Based on profiling data, these functions consume the most computation time:

1. **`_interpolate()`** (~40% of CPU time)
   - Called thousands of times per frame
   - 4-8 calls per window pixel per iteration
   - Simple but frequent: bilinear interpolation

2. **`_convolveImageVert()`** (~30% of CPU time)
   - Called for every gradient and smoothing operation
   - Processes entire image multiple times
   - Nested loops over image dimensions

3. **`_convolveImageHoriz()`** (~10% of CPU time)
   - Similar to vertical convolution
   - Always paired with vertical pass

4. **`_minEigenvalue()`** (during feature selection)
   - Called once per potential feature location
   - Includes expensive `sqrt()` operation

5. **`_trackFeature()`** (orchestration overhead)
   - Coordinates the iterative tracking
   - Calls many sub-functions

These are the primary targets for optimization (which is why GPU versions exist for interpolation and convolution!).

---

## The Complete Workflow

Let's follow a feature through the entire process:

### Frame 0 (First Image):
1. **Load image** (320×240 grayscale image)
   - Function: `pgmReadFile()` (in `pnmio.c`)
   - Optionally: `resizeImage()` in example3 (for heavier workload)

2. **Initialize tracking**
   - `KLTCreateTrackingContext()` sets up parameters
   - `KLTCreateFeatureList(nFeatures)` allocates feature array

3. **Feature Selection Process**
   - `KLTSelectGoodFeatures(tc, img, ncols, nrows, fl)` is called
   
   **Internal flow:**
   - a. `_KLTToFloatImage()` converts byte image to float
   - b. `_KLTComputeSmoothedImage()` applies Gaussian blur
      - `_convolveSeparate()` calls:
        - `_convolveImageHoriz()` - blur horizontally
        - `_convolveImageVert()` - blur vertically
   - c. `_KLTComputeGradients()` computes brightness changes
      - Creates gradx: derivative in x, smoothed in y
      - Creates grady: smoothed in x, derivative in y
      - Each uses `_convolveSeparate()` with different kernel pairs
   
4. **Evaluate every potential feature location**
   - For each pixel (skipping borders):
     - Extract 7×7 window of gradients around it
     - Calculate: `gxx = Σ(gradx²)`, `gxy = Σ(gradx·grady)`, `gyy = Σ(grady²)`
     - Call `_minEigenvalue(gxx, gxy, gyy)` → get quality score
     - Store `[x, y, score]` in pointlist
   
5. **Select best features**
   - `_sortPointList()` sorts by score (best first)
   - `_enforceMinimumDistance()` picks top features:
     - Go through sorted list
     - Keep feature if: (a) high enough score, (b) not too close to already-selected feature
     - `_fillFeaturemap()` marks nearby area as occupied
   
6. **Store initial positions**
   - Feature list now has 150 features with (x, y) positions
   - Each feature has `val = 0` (valid)

### Frame 1 (Next Image):
1. **Load new image**
   - Function: `pgmReadFile()` for img1.pgm
   - Optionally resized

2. **Build image pyramids**
   - For previous frame (Frame 0):
     - If `sequentialMode` enabled: Use cached pyramid from `tc->pyramid_last`
     - Otherwise: Build fresh pyramid
   
   - For current frame (Frame 1):
     - `_KLTToFloatImage()` converts to float
     - `_KLTComputeSmoothedImage()` smooths
     - `_KLTCreatePyramid()` allocates pyramid structure
     - `_KLTComputePyramid()` fills it:
       - Level 0: Copy smoothed image
       - Level 1: Smooth level 0, subsample by factor (e.g., 4)
       - Level 2: Smooth level 1, subsample again
     - `_KLTComputeGradients()` computes gradx/grady for each level
   
3. **Track each of the 150 features**
   - `KLTTrackFeatures(tc, img0, img1, ncols, nrows, fl)` is called
   
   **For feature #0:**
   - Starting position: `x₀ = 45.0, y₀ = 123.0` (from Frame 0)
   
   - **Level 2 (coarsest, e.g., 1/16 resolution):**
     - Scale position: `x = 45/16 = 2.8125, y = 123/16 = 7.6875`
     - Call `_trackFeature()` at this level:
       - **Iteration 1:**
         - `_computeIntensityDifference()`: Compare 7×7 windows
           - For each of 49 pixels: call `_interpolate()` twice (Frame 0 and Frame 1)
           - Total: 98 interpolations → compute differences
         - `_computeGradientSum()`: Combine gradients
           - For each of 49 pixels: call `_interpolate()` 4 times (gradx0, gradx1, grady0, grady1)
           - Total: 196 interpolations → sum gradients
         - `_compute2by2GradientMatrix()`: Build structure tensor from gradient window
         - `_compute2by1ErrorVector()`: Build error vector from differences
         - `_solveEquation()`: Solve for `(dx, dy)` displacement
         - Update: `x += dx, y += dy`
       - **Iteration 2-10:** Repeat until `|dx| < 0.1` and `|dy| < 0.1`
       - Suppose converges at iteration 3: `x = 3.1, y = 7.9`
     
   - **Level 1 (medium, e.g., 1/4 resolution):**
     - Scale up: `x = 3.1 × 4 = 12.4, y = 7.9 × 4 = 31.6`
     - Call `_trackFeature()` at this level:
       - Iterate again (similar process, but on finer image)
       - Suppose converges: `x = 12.6, y = 31.8`
     
   - **Level 0 (finest, full resolution):**
     - Scale up: `x = 12.6 × 4 = 50.4, y = 31.8 × 4 = 127.2`
     - Call `_trackFeature()` at this level:
       - Iterate for precise sub-pixel location
       - Suppose converges: `x = 50.2, y = 127.5`
     
   - **Final validation:**
     - `_outOfBounds()` check: Is `(50.2, 127.5)` inside image? Yes
     - `_computeIntensityDifference()` one last time
     - `_sumAbsFloatWindow()` calculates total difference (residue)
     - If residue < `max_residue`: Feature successfully tracked!
   
   - **Update feature list:**
     - `fl->feature[0]->x = 50.2`
     - `fl->feature[0]->y = 127.5`
     - `fl->feature[0]->val = KLT_TRACKED` (0)
   
   **For feature #1, #2, ... #149:** Repeat the same process
   
4. **Results after Frame 1:**
   - Feature #0: TRACKED, moved from (45.0, 123.0) to (50.2, 127.5)
   - Feature #7: LOST (KLT_OOB), moved outside image boundary
   - Feature #23: LOST (KLT_LARGE_RESIDUE), poor match quality
   - Feature #42: TRACKED, moved from (201.5, 87.3) to (199.8, 89.1)
   - ... and so on for all 150 features
   
5. **Optional: Replace lost features**
   - `KLTReplaceLostFeatures(tc, img1, ncols, nrows, fl)`
   - Runs feature selection again, but only fills slots of lost features
   - Ensures you always have ~150 features to track

### Frames 2, 3, 4...:
- Repeat the tracking process
- Each frame uses the previous frame's feature positions as starting points
- Features accumulate drift over time (small errors compound)
- More features get lost (occlusion, out of bounds, poor matching)
- If replacing: New features are continuously added

### Storage and Visualization:
- **Store results:**
  - `KLTStoreFeatureList(fl, ft, frameNumber)` saves positions to feature table
  
- **Visualize:**
  - `KLTWriteFeatureListToPPM(fl, img, ncols, nrows, "feat1.ppm")` creates image with feature markers
  
- **Export:**
  - `KLTWriteFeatureTable(ft, "features.txt", "%5.1f")` writes all tracking data to text file

### End Result:
After processing 10 frames, you have:
- A feature table with positions of each feature in each frame
- Visualization images showing tracked features
- Statistics on how many features were successfully tracked through all frames

### Function Call Summary Per Frame:

**Feature Selection (Frame 0 only):**
- 1× `KLTSelectGoodFeatures()`
  - 1× image conversion and smoothing
  - 1× gradient computation (2 convolutions)
  - ~300,000 eigenvalue calculations (for 320×240 image)
  - 1 sort operation
  - 1 spacing enforcement

**Feature Tracking (Frames 1+):**
- 1× `KLTTrackFeatures()`
  - 2-3× pyramid builds (6-9 levels × 2 images)
  - 2-3× gradient computations per level
  - 150 features × 3 pyramid levels × ~5 iterations = ~2,250 tracking iterations
    - Each iteration: ~300 interpolations
    - **Total: ~675,000 interpolations per frame!**
  - This is why `_interpolate()` dominates CPU time

### Data Flow Visualization:

```
Frame 0 Image (bytes)
    ↓ [_KLTToFloatImage]
Float Image
    ↓ [_KLTComputeSmoothedImage]
Smoothed Image
    ↓ [_KLTComputeGradients]
Gradients (gradx, grady)
    ↓ [_minEigenvalue × many]
Eigenvalue Map
    ↓ [_sortPointList]
Sorted Feature Candidates
    ↓ [_enforceMinimumDistance]
Feature List (150 features)

---

Frame 1 Image (bytes)
    ↓ [_KLTToFloatImage]
Float Image
    ↓ [_KLTComputePyramid]
Image Pyramid (3 levels)
    ↓ [_KLTComputeGradients per level]
Gradient Pyramids
    ↓ [_trackFeature for each feature, each level]
    │ (uses _interpolate, _solve, etc.)
Updated Feature Positions

---

Frame 2, 3, 4... (repeat tracking)
```

---

## Common Parameters

These are the knobs you can turn to adjust KLT's behavior:

- **`window_width`, `window_height`** (default: 7×7): Size of the template window
  - Larger = more stable but slower
  - Smaller = faster but less reliable

- **`nPyramidLevels`** (default: 2-3): How many pyramid levels
  - More levels = can track faster motion
  - Fewer levels = faster but limited to small motion

- **`min_eigenvalue`** (default: 1): Minimum cornerness score to be a feature
  - Higher = only very strong corners
  - Lower = accept weaker features

- **`min_displacement`** (default: 0.1): When to stop iterating
  - When position changes by less than this, we've converged

- **`max_iterations`** (default: 10): Maximum refinement steps
  - More = more accurate but slower
  - Less = faster but might not converge

- **`max_residue`** (default: 10.0): How different windows can be
  - Lower = stricter matching, more features marked as lost
  - Higher = more lenient, might track wrong features

---

## Why Features Get Lost

A feature can be marked as lost for several reasons:

1. **Out of Bounds (OOB)**: The feature moved outside the image frame
2. **Small Determinant**: The math became unstable (usually means the feature is in a flat region now)
3. **Large Residue**: The window in Frame 2 looks too different from Frame 1 (probably wrong match)
4. **Max Iterations**: Couldn't converge within the iteration limit
5. **Occlusion** (hidden): Something covered the feature (the algorithm doesn't explicitly detect this, but it shows up as large residue)

---

## Real-World Analogy

Imagine you're watching a nature documentary about birds:

### Feature Selection = Picking Which Birds to Track
- You don't try to track the sky (flat, boring)
- You don't track blurry distant trees (no detail)
- **You track the birds!** (high contrast, distinctive features)
- You pick specific birds spaced apart, not all clustered together

### Feature Tracking = Following Your Bird Across Frames
- You remember what your bird looked like in the last frame (its pattern, colors)
- You expect it to be nearby (it didn't teleport)
- You look around that area for something matching your memory
- You refine your guess until you're confident you found the right bird
- If the bird flies behind a tree, you lose track (occlusion)
- If the bird flies out of frame, you lose track (out of bounds)

---

## Summary

**KLT in one sentence**: KLT finds distinctive corner points in the first image and efficiently follows them across subsequent images by matching small windows using iterative optimization on multiple image scales.

**Key Strengths**:
- Fast and efficient
- Works well for small to medium motion
- Good for real-time applications

**Key Limitations**:
- Assumes brightness doesn't change much (fails with lighting changes)
- Assumes small motion between frames (can't handle jumps)
- Features get lost over time (occlusion, going out of frame)
- No mechanism to recover lost features (unless you re-detect)

---

## What This Implementation Does

Looking at `example3.c`, this specific implementation:
1. Loads 10 images (`img0.pgm` through `img9.pgm`)
2. Upscales them 2× to make computation heavier (for benchmarking)
3. Selects 300 good features in the first image
4. Tracks them through all subsequent images
5. Saves visualization images showing tracked features
6. Writes results to a text file

The code has been optimized with CUDA (GPU acceleration) for three major bottlenecks:
- Feature selection (eigenvalue calculation)
- Interpolation (calculating sub-pixel values)
- Convolution (smoothing and gradients)

But the core algorithm logic remains the same!

---

## Conclusion

KLT is like having a very attentive assistant who:
1. Picks out interesting landmarks in a scene
2. Remembers what they look like
3. Carefully follows them as things move
4. Tells you where each landmark moved to

It's been used for decades in computer vision because it's simple, fast, and effective. Now you understand how it works under the hood!
