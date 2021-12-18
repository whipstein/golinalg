package lin

import (
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// dqrt17 computes the ratio
//
//    || R'*op(A) ||/(||A||*alpha*max(M,N,NRHS)*eps)
//
// where R = op(A)*X - B, op(A) is A or A', and
//
//    alpha = ||B|| if IRESID = 1 (zero-residual problem)
//    alpha = ||R|| if IRESID = 2 (otherwise).
func dqrt17(trans mat.MatTrans, iresid, m, n, nrhs int, a, x, b, c *mat.Matrix, work *mat.Vector, lwork int) (dqrt17Return float64) {
	var err2, norma, normb, normrs, one, smlnum, zero float64
	var iscl, ncols, nrows int
	var err error

	rwork := vf(1)

	zero = 0.0
	one = 1.0

	dqrt17Return = zero

	if trans == NoTrans {
		nrows = m
		ncols = n
	} else if trans == Trans {
		nrows = n
		ncols = m
	} else {
		gltest.Xerbla("dqrt17", 1)
		return
	}

	if lwork < ncols*nrhs {
		gltest.Xerbla("dqrt17", 13)
		return
	}

	if m <= 0 || n <= 0 || nrhs <= 0 {
		return
	}

	norma = golapack.Dlange('O', m, n, a, rwork)
	smlnum = golapack.Dlamch(SafeMinimum) / golapack.Dlamch(Precision)
	// bignum = one / smlnum
	iscl = 0

	//     compute residual and scale it
	golapack.Dlacpy(Full, nrows, nrhs, b, c)
	if err = c.Gemm(trans, NoTrans, nrows, nrhs, ncols, -one, a, x, one); err != nil {
		panic(err)
	}
	normrs = golapack.Dlange('M', nrows, nrhs, c, rwork)
	if normrs > smlnum {
		iscl = 1
		if err = golapack.Dlascl('G', 0, 0, normrs, one, nrows, nrhs, c); err != nil {
			panic(err)
		}
	}

	//     compute R'*A
	if err = work.Matrix(nrhs, opts).Gemm(Trans, trans, nrhs, ncols, nrows, one, c, a, zero); err != nil {
		panic(err)
	}

	//     compute and properly scale error
	err2 = golapack.Dlange('O', nrhs, ncols, work.Matrix(nrhs, opts), rwork)
	if norma != zero {
		err2 /= norma
	}

	if iscl == 1 {
		err2 *= normrs
	}

	if iresid == 1 {
		normb = golapack.Dlange('O', nrows, nrhs, b, rwork)
		if normb != zero {
			err2 /= normb
		}
	} else {
		if normrs != zero {
			err2 /= normrs
		}
	}

	dqrt17Return = err2 / (golapack.Dlamch(Epsilon) * float64(max(m, n, nrhs)))
	return
}
