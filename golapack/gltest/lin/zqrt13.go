package lin

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// Zqrt13 generates a full-rank matrix that may be scaled to have large
// or small norm.
func Zqrt13(scale, m, n *int, a *mat.CMatrix, lda *int, norma *float64, iseed *[]int) {
	var bignum, one, smlnum float64
	var info, j int

	dummy := vf(1)

	one = 1.0

	if (*m) <= 0 || (*n) <= 0 {
		return
	}

	//     benign matrix
	for j = 1; j <= (*n); j++ {
		golapack.Zlarnv(func() *int { y := 2; return &y }(), iseed, m, a.CVector(0, j-1))
		if j <= (*m) {
			a.Set(j-1, j-1, a.Get(j-1, j-1)+complex(math.Copysign(goblas.Dzasum(*m, a.CVector(0, j-1), 1), a.GetRe(j-1, j-1)), 0))
		}
	}

	//     scaled versions
	if (*scale) != 1 {
		(*norma) = golapack.Zlange('M', m, n, a, lda, dummy)
		smlnum = golapack.Dlamch(SafeMinimum)
		bignum = one / smlnum
		golapack.Dlabad(&smlnum, &bignum)
		smlnum = smlnum / golapack.Dlamch(Epsilon)
		bignum = one / smlnum

		if (*scale) == 2 {
			//           matrix scaled up
			golapack.Zlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), norma, &bignum, m, n, a, lda, &info)
		} else if (*scale) == 3 {
			//           matrix scaled down
			golapack.Zlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), norma, &smlnum, m, n, a, lda, &info)
		}
	}

	(*norma) = golapack.Zlange('O', m, n, a, lda, dummy)
}
