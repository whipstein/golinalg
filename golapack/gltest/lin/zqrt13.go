package lin

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// zqrt13 generates a full-rank matrix that may be scaled to have large
// or small norm.
func zqrt13(scale, m, n int, a *mat.CMatrix, iseed *[]int) (norma float64) {
	var bignum, one, smlnum float64
	var j int
	var err error

	dummy := vf(1)

	one = 1.0

	if m <= 0 || n <= 0 {
		return
	}

	//     benign matrix
	for j = 1; j <= n; j++ {
		golapack.Zlarnv(2, iseed, m, a.CVector(0, j-1))
		if j <= m {
			a.Set(j-1, j-1, a.Get(j-1, j-1)+complex(math.Copysign(goblas.Dzasum(m, a.CVector(0, j-1, 1)), a.GetRe(j-1, j-1)), 0))
		}
	}

	//     scaled versions
	if scale != 1 {
		norma = golapack.Zlange('M', m, n, a, dummy)
		smlnum = golapack.Dlamch(SafeMinimum)
		bignum = one / smlnum
		smlnum, bignum = golapack.Dlabad(smlnum, bignum)
		smlnum = smlnum / golapack.Dlamch(Epsilon)
		bignum = one / smlnum

		if scale == 2 {
			//           matrix scaled up
			if err = golapack.Zlascl('G', 0, 0, norma, bignum, m, n, a); err != nil {
				panic(err)
			}
		} else if scale == 3 {
			//           matrix scaled down
			if err = golapack.Zlascl('G', 0, 0, norma, smlnum, m, n, a); err != nil {
				panic(err)
			}
		}
	}

	norma = golapack.Zlange('O', m, n, a, dummy)

	return
}
