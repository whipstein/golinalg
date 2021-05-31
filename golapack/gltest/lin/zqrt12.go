package lin

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zqrt12 computes the singular values `svlues' of the upper trapezoid
// of A(1:M,1:N) and returns the ratio
//
//      || s - svlues||/(||svlues||*eps*maxint(M,N))
func Zqrt12(m, n *int, a *mat.CMatrix, lda *int, s *mat.Vector, work *mat.CVector, lwork *int, rwork *mat.Vector) (zqrt12Return float64) {
	var anrm, bignum, nrmsvl, one, smlnum, zero float64
	var i, info, iscl, j, mn int

	dummy := vf(1)

	zero = 0.0
	one = 1.0

	zqrt12Return = zero

	//     Test that enough workspace is supplied
	if (*lwork) < (*m)*(*n)+2*minint(*m, *n)+maxint(*m, *n) {
		gltest.Xerbla([]byte("ZQRT12"), 7)
		return
	}

	//     Quick return if possible
	mn = minint(*m, *n)
	if mn <= int(zero) {
		return
	}

	nrmsvl = goblas.Dnrm2(&mn, s, func() *int { y := 1; return &y }())

	//     Copy upper triangle of A into work
	golapack.Zlaset('F', m, n, toPtrc128(complex(zero, 0)), toPtrc128(complex(zero, 0)), work.CMatrix(*m, opts), m)
	for j = 1; j <= (*n); j++ {
		for i = 1; i <= minint(j, *m); i++ {
			work.Set((j-1)*(*m)+i-1, a.Get(i-1, j-1))
		}
	}

	//     Get machine parameters
	smlnum = golapack.Dlamch(SafeMinimum) / golapack.Dlamch(Precision)
	bignum = one / smlnum
	golapack.Dlabad(&smlnum, &bignum)

	//     Scale work if maxint entry outside range [SMLNUM,BIGNUM]
	anrm = golapack.Zlange('M', m, n, work.CMatrix(*m, opts), m, dummy)
	iscl = 0
	if anrm > zero && anrm < smlnum {
		//        Scale matrix norm up to SMLNUM
		golapack.Zlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &anrm, &smlnum, m, n, work.CMatrix(*m, opts), m, &info)
		iscl = 1
	} else if anrm > bignum {
		//        Scale matrix norm down to BIGNUM
		golapack.Zlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &anrm, &bignum, m, n, work.CMatrix(*m, opts), m, &info)
		iscl = 1
	}

	if anrm != zero {
		//        Compute SVD of work
		golapack.Zgebd2(m, n, work.CMatrix(*m, opts), m, rwork.Off(0), rwork.Off(mn+1-1), work.Off((*m)*(*n)+1-1), work.Off((*m)*(*n)+mn+1-1), work.Off((*m)*(*n)+2*mn+1-1), &info)
		golapack.Dbdsqr('U', &mn, func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), rwork.Off(0), rwork.Off(mn+1-1), dummy.Matrix(mn, opts), &mn, dummy.Matrix(1, opts), func() *int { y := 1; return &y }(), dummy.Matrix(mn, opts), &mn, rwork.Off(2*mn+1-1), &info)

		if iscl == 1 {
			if anrm > bignum {
				golapack.Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &bignum, &anrm, &mn, func() *int { y := 1; return &y }(), rwork.Matrix(mn, opts), &mn, &info)
			}
			if anrm < smlnum {
				golapack.Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &smlnum, &anrm, &mn, func() *int { y := 1; return &y }(), rwork.Matrix(mn, opts), &mn, &info)
			}
		}

	} else {

		for i = 1; i <= mn; i++ {
			rwork.Set(i-1, zero)
		}
	}

	//     Compare s and singular values of work
	goblas.Daxpy(&mn, toPtrf64(-one), s, func() *int { y := 1; return &y }(), rwork.Off(0), func() *int { y := 1; return &y }())
	zqrt12Return = goblas.Dasum(&mn, rwork.Off(0), func() *int { y := 1; return &y }()) / (golapack.Dlamch(Epsilon) * float64(maxint(*m, *n)))
	if nrmsvl != zero {
		zqrt12Return = zqrt12Return / nrmsvl
	}
	return
}
