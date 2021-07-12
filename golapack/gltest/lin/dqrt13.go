package lin

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// Dqrt13 generates a full-rank matrix that may be scaled to have large
// or small norm.
func Dqrt13(scale, m, n *int, a *mat.Matrix, lda *int, norma *float64, iseed *[]int) {
	var bignum, one, smlnum float64
	var info, j int

	dummy := vf(1)

	one = 1.0

	if (*m) <= 0 || (*n) <= 0 {
		return
	}

	//     benign matrix
	for j = 1; j <= (*n); j++ {
		golapack.Dlarnv(func() *int { y := 2; return &y }(), iseed, m, a.Vector(0, j-1))
		if j <= (*m) {
			a.Set(j-1, j-1, a.Get(j-1, j-1)+math.Copysign(goblas.Dasum(*m, a.Vector(0, j-1, 1)), a.Get(j-1, j-1)))
		}
	}

	//     scaled versions
	if (*scale) != 1 {
		(*norma) = golapack.Dlange('M', m, n, a, lda, dummy)
		smlnum = golapack.Dlamch(SafeMinimum)
		bignum = one / smlnum
		golapack.Dlabad(&smlnum, &bignum)
		smlnum = smlnum / golapack.Dlamch(Epsilon)
		bignum = one / smlnum

		if (*scale) == 2 {
			//           matrix scaled up
			golapack.Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), norma, &bignum, m, n, a, lda, &info)
		} else if (*scale) == 3 {
			//           matrix scaled down
			golapack.Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), norma, &smlnum, m, n, a, lda, &info)
		}
	}

	(*norma) = golapack.Dlange('O', m, n, a, lda, dummy)
}
