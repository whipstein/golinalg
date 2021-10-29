package lin

import (
	"fmt"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// zqrt12 computes the singular values `svlues' of the upper trapezoid
// of A(1:M,1:N) and returns the ratio
//
//      || s - svlues||/(||svlues||*eps*max(M,N))
func zqrt12(m, n int, a *mat.CMatrix, s *mat.Vector, work *mat.CVector, lwork int, rwork *mat.Vector) (zqrt12Return float64) {
	var anrm, bignum, nrmsvl, one, smlnum, zero float64
	var i, iscl, j, mn int
	var err error

	dummy := vf(1)

	zero = 0.0
	one = 1.0

	zqrt12Return = zero

	//     Test that enough workspace is supplied
	if lwork < m*n+2*min(m, n)+max(m, n) {
		err = fmt.Errorf("lwork < mn+2*min(m, n)+max(m, n): lwork=%v, m=%v, n=%v", lwork, m, n)
		gltest.Xerbla2("zqrt12", err)
		return
	}

	//     Quick return if possible
	mn = min(m, n)
	if mn <= int(zero) {
		return
	}

	nrmsvl = goblas.Dnrm2(mn, s.Off(0, 1))

	//     Copy upper triangle of A into work
	golapack.Zlaset(Full, m, n, complex(zero, 0), complex(zero, 0), work.CMatrix(m, opts))
	for j = 1; j <= n; j++ {
		for i = 1; i <= min(j, m); i++ {
			work.Set((j-1)*m+i-1, a.Get(i-1, j-1))
		}
	}

	//     Get machine parameters
	smlnum = golapack.Dlamch(SafeMinimum) / golapack.Dlamch(Precision)
	bignum = one / smlnum
	smlnum, bignum = golapack.Dlabad(smlnum, bignum)

	//     Scale work if max entry outside range [SMLNUM,BIGNUM]
	anrm = golapack.Zlange('M', m, n, work.CMatrix(m, opts), dummy)
	iscl = 0
	if anrm > zero && anrm < smlnum {
		//        Scale matrix norm up to SMLNUM
		if err = golapack.Zlascl('G', 0, 0, anrm, smlnum, m, n, work.CMatrix(m, opts)); err != nil {
			panic(err)
		}
		iscl = 1
	} else if anrm > bignum {
		//        Scale matrix norm down to BIGNUM
		if err = golapack.Zlascl('G', 0, 0, anrm, bignum, m, n, work.CMatrix(m, opts)); err != nil {
			panic(err)
		}
		iscl = 1
	}

	if anrm != zero {
		//        Compute SVD of work
		if err = golapack.Zgebd2(m, n, work.CMatrix(m, opts), rwork.Off(0), rwork.Off(mn), work.Off(m*n), work.Off(m*n+mn), work.Off(m*n+2*mn)); err != nil {
			panic(err)
		}
		if _, err = golapack.Dbdsqr(Upper, mn, 0, 0, 0, rwork.Off(0), rwork.Off(mn), dummy.Matrix(mn, opts), dummy.Matrix(1, opts), dummy.Matrix(mn, opts), rwork.Off(2*mn)); err != nil {
			panic(err)
		}

		if iscl == 1 {
			if anrm > bignum {
				if err = golapack.Dlascl('G', 0, 0, bignum, anrm, mn, 1, rwork.Matrix(mn, opts)); err != nil {
					panic(err)
				}
			}
			if anrm < smlnum {
				if err = golapack.Dlascl('G', 0, 0, smlnum, anrm, mn, 1, rwork.Matrix(mn, opts)); err != nil {
					panic(err)
				}
			}
		}

	} else {

		for i = 1; i <= mn; i++ {
			rwork.Set(i-1, zero)
		}
	}

	//     Compare s and singular values of work
	goblas.Daxpy(mn, -one, s.Off(0, 1), rwork.Off(0, 1))
	zqrt12Return = goblas.Dasum(mn, rwork.Off(0, 1)) / (golapack.Dlamch(Epsilon) * float64(max(m, n)))
	if nrmsvl != zero {
		zqrt12Return = zqrt12Return / nrmsvl
	}
	return
}
