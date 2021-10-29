package lin

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// dqrt13 generates a full-rank matrix that may be scaled to have large
// or small norm.
func dqrt13(scale, m, n int, a *mat.Matrix, iseed []int) (float64, []int) {
	var bignum, norma, one, smlnum float64
	var j int
	var err error

	dummy := vf(1)

	one = 1.0

	if m <= 0 || n <= 0 {
		return norma, iseed
	}

	//     benign matrix
	for j = 1; j <= n; j++ {
		golapack.Dlarnv(2, &iseed, m, a.Vector(0, j-1))
		if j <= m {
			a.Set(j-1, j-1, a.Get(j-1, j-1)+math.Copysign(goblas.Dasum(m, a.Vector(0, j-1, 1)), a.Get(j-1, j-1)))
		}
	}

	//     scaled versions
	if scale != 1 {
		norma = golapack.Dlange('M', m, n, a, dummy)
		smlnum = golapack.Dlamch(SafeMinimum)
		bignum = one / smlnum
		smlnum, bignum = golapack.Dlabad(smlnum, bignum)
		smlnum = smlnum / golapack.Dlamch(Epsilon)
		bignum = one / smlnum

		if scale == 2 {
			//           matrix scaled up
			if err = golapack.Dlascl('G', 0, 0, norma, bignum, m, n, a); err != nil {
				panic(err)
			}
		} else if scale == 3 {
			//           matrix scaled down
			if err = golapack.Dlascl('G', 0, 0, norma, smlnum, m, n, a); err != nil {
				panic(err)
			}
		}
	}

	norma = golapack.Dlange('O', m, n, a, dummy)

	return norma, iseed
}
