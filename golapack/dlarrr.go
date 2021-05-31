package golapack

import (
	"math"

	"github.com/whipstein/golinalg/mat"
)

// Dlarrr Perform tests to decide whether the symmetric tridiagonal matrix T
// warrants expensive computations which guarantee high relative accuracy
// in the eigenvalues.
func Dlarrr(n *int, d, e *mat.Vector, info *int) {
	var yesrel bool
	var eps, offdig, offdig2, relcond, rmin, safmin, smlnum, tmp, tmp2, zero float64
	var i int

	zero = 0.0
	relcond = 0.999

	//     Quick return if possible
	if (*n) <= 0 {
		(*info) = 0
		return
	}

	//     As a default, do NOT go for relative-accuracy preserving computations.
	(*info) = 1
	safmin = Dlamch(SafeMinimum)
	eps = Dlamch(Precision)
	smlnum = safmin / eps
	rmin = math.Sqrt(smlnum)
	//     Tests for relative accuracy
	//
	//     Test for scaled diagonal dominance
	//     Scale the diagonal entries to one and check whether the sum of the
	//     off-diagonals is less than one
	//
	//     The sdd relative error bounds have a 1/(1- 2*x) factor in them,
	//     x = max(OFFDIG + OFFDIG2), so when x is close to 1/2, no relative
	//     accuracy is promised.  In the notation of the code fragment below,
	//     1/(1 - (OFFDIG + OFFDIG2)) is the condition number.
	//     We don't think it is worth going into "sdd mode" unless the relative
	//     condition number is reasonable, not 1/macheps.
	//     The threshold should be compatible with other thresholds used in the
	//     code. We set  OFFDIG + OFFDIG2 <= .999 =: RELCOND, it corresponds
	//     to losing at most 3 decimal digits: 1 / (1 - (OFFDIG + OFFDIG2)) <= 1000
	//     instead of the current OFFDIG + OFFDIG2 < 1
	yesrel = true
	offdig = zero
	tmp = math.Sqrt(math.Abs(d.Get(0)))
	if tmp < rmin {
		yesrel = false
	}
	if !yesrel {
		goto label11
	}
	for i = 2; i <= (*n); i++ {
		tmp2 = math.Sqrt(math.Abs(d.Get(i - 1)))
		if tmp2 < rmin {
			yesrel = false
		}
		if !yesrel {
			goto label11
		}
		offdig2 = math.Abs(e.Get(i-1-1)) / (tmp * tmp2)
		if offdig+offdig2 >= relcond {
			yesrel = false
		}
		if !yesrel {
			goto label11
		}
		tmp = tmp2
		offdig = offdig2
	}
label11:
	;
	if yesrel {
		(*info) = 0
		return
	} else {
	}

	//     *** MORE TO BE IMPLEMENTED ***
	//
	//
	//     Test if the lower bidiagonal matrix L from T = L D L^T
	//     (zero shift facto) is well conditioned
	//
	//
	//     Test if the upper bidiagonal matrix U from T = U D U^T
	//     (zero shift facto) is well conditioned.
	//     In this case, the matrix needs to be flipped and, at the end
	//     of the eigenvector computation, the flip needs to be applied
	//     to the computed eigenvectors (and the support)
}
