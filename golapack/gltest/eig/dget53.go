package eig

import (
	"math"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// dget53 checks the generalized eigenvalues computed by DLAG2.
//
// The basic test for an eigenvalue is:
//
//                              | det( s A - w B ) |
//     RESULT =  ---------------------------------------------------
//               ulp math.Max( s norm(A), |w| norm(B) )*norm( s A - w B )
//
// Two "safety checks" are performed:
//
// (1)  ulp*math.Max( s*norm(A), |w|*norm(B) )  must be at least
//      safe_minimum.  This insures that the test performed is
//      not essentially  det(0*A + 0*B)=0.
//
// (2)  s*norm(A) + |w|*norm(B) must be less than 1/safe_minimum.
//      This insures that  s*A - w*B  will not overflow.
//
// If these tests are not passed, then  s  and  w  are scaled and
// tested anyway, if this is possible.
func dget53(a, b *mat.Matrix, scale, wr, wi float64) (result float64, info int) {
	var absw, anorm, bnorm, ci11, ci12, ci22, cnorm, cr11, cr12, cr21, cr22, cscale, deti, detr, one, s1, safmin, scales, sigmin, temp, ulp, wis, wrs, zero float64

	zero = 0.0
	one = 1.0

	//     Initialize
	info = 0
	result = zero
	scales = scale
	wrs = wr
	wis = wi

	//     Machine constants and norms
	safmin = golapack.Dlamch(SafeMinimum)
	ulp = golapack.Dlamch(Epsilon) * golapack.Dlamch(Base)
	absw = math.Abs(wrs) + math.Abs(wis)
	anorm = math.Max(math.Abs(a.Get(0, 0))+math.Abs(a.Get(1, 0)), math.Max(math.Abs(a.Get(0, 1))+math.Abs(a.Get(1, 1)), safmin))
	bnorm = math.Max(math.Abs(b.Get(0, 0)), math.Max(math.Abs(b.Get(0, 1))+math.Abs(b.Get(1, 1)), safmin))

	//     Check for possible overflow.
	temp = (safmin*bnorm)*absw + (safmin*anorm)*scales
	if temp >= one {
		//        Scale down to avoid overflow
		info = 1
		temp = one / temp
		scales *= temp
		wrs *= temp
		wis *= temp
		absw = math.Abs(wrs) + math.Abs(wis)
	}
	s1 = math.Max(ulp*math.Max(scales*anorm, absw*bnorm), safmin*math.Max(scales, absw))

	//     Check for W and SCALE essentially zero.
	if s1 < safmin {
		info = 2
		if scales < safmin && absw < safmin {
			info = 3
			result = one / ulp
			return
		}

		//        Scale up to avoid underflow
		temp = one / math.Max(scales*anorm+absw*bnorm, safmin)
		scales *= temp
		wrs *= temp
		wis *= temp
		absw = math.Abs(wrs) + math.Abs(wis)
		s1 = math.Max(ulp*math.Max(scales*anorm, absw*bnorm), safmin*math.Max(scales, absw))
		if s1 < safmin {
			info = 3
			result = one / ulp
			return
		}
	}

	//     Compute C = s A - w B
	cr11 = scales*a.Get(0, 0) - wrs*b.Get(0, 0)
	ci11 = -wis * b.Get(0, 0)
	cr21 = scales * a.Get(1, 0)
	cr12 = scales*a.Get(0, 1) - wrs*b.Get(0, 1)
	ci12 = -wis * b.Get(0, 1)
	cr22 = scales*a.Get(1, 1) - wrs*b.Get(1, 1)
	ci22 = -wis * b.Get(1, 1)

	//     Compute the smallest singular value of s A - w B:
	//
	//                 |det( s A - w B )|
	//     sigma_min = ------------------
	//                 norm( s A - w B )
	cnorm = math.Max(math.Abs(cr11)+math.Abs(ci11)+math.Abs(cr21), math.Max(math.Abs(cr12)+math.Abs(ci12)+math.Abs(cr22)+math.Abs(ci22), safmin))
	cscale = one / math.Sqrt(cnorm)
	detr = (cscale*cr11)*(cscale*cr22) - (cscale*ci11)*(cscale*ci22) - (cscale*cr12)*(cscale*cr21)
	deti = (cscale*cr11)*(cscale*ci22) + (cscale*ci11)*(cscale*cr22) - (cscale*ci12)*(cscale*cr21)
	sigmin = math.Abs(detr) + math.Abs(deti)
	result = sigmin / s1

	return
}
