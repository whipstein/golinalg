package lin

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dqrt12 computes the singular values `svlues' of the upper trapezoid
// of A(1:M,1:N) and returns the ratio
//
//      || s - svlues||/(||svlues||*eps*max(M,N))
func Dqrt12(m, n *int, a *mat.Matrix, lda *int, s, work *mat.Vector, lwork *int) (dqrt12Return float64) {
	var anrm, bignum, nrmsvl, one, smlnum, zero float64
	var i, info, iscl, j, mn int

	dummy := vf(1)

	zero = 0.0
	one = 1.0

	dqrt12Return = zero

	//     Test that enough workspace is supplied
	if (*lwork) < max((*m)*(*n)+4*min(*m, *n)+max(*m, *n), (*m)*(*n)+2*min(*m, *n)+4*(*n)) {
		gltest.Xerbla([]byte("DQRT12"), 7)
		return
	}

	//     Quick return if possible
	mn = min(*m, *n)
	if mn <= int(zero) {
		return
	}

	nrmsvl = goblas.Dnrm2(mn, s.Off(0, 1))

	//     Copy upper triangle of A into work
	golapack.Dlaset('F', m, n, &zero, &zero, work.Matrix(*m, opts), m)
	for j = 1; j <= (*n); j++ {
		for i = 1; i <= min(j, *m); i++ {
			work.Set((j-1)*(*m)+i-1, a.Get(i-1, j-1))
		}
	}

	//     Get machine parameters
	smlnum = golapack.Dlamch(SafeMinimum) / golapack.Dlamch(Precision)
	bignum = one / smlnum
	golapack.Dlabad(&smlnum, &bignum)

	//     Scale work if max entry outside range [SMLNUM,BIGNUM]
	anrm = golapack.Dlange('M', m, n, work.Matrix(*m, opts), m, dummy)
	iscl = 0
	if anrm > zero && anrm < smlnum {
		//        Scale matrix norm up to SMLNUM
		golapack.Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &anrm, &smlnum, m, n, work.Matrix(*m, opts), m, &info)
		iscl = 1
	} else if anrm > bignum {
		//        Scale matrix norm down to BIGNUM
		golapack.Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &anrm, &bignum, m, n, work.Matrix(*m, opts), m, &info)
		iscl = 1
	}

	if anrm != zero {
		//        Compute SVD of work
		golapack.Dgebd2(m, n, work.Matrix(*m, opts), m, work.Off((*m)*(*n)), work.Off((*m)*(*n)+mn), work.Off((*m)*(*n)+2*mn), work.Off((*m)*(*n)+3*mn), work.Off((*m)*(*n)+4*mn), &info)
		golapack.Dbdsqr('U', &mn, func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), work.Off((*m)*(*n)), work.Off((*m)*(*n)+mn), dummy.Matrix(1, opts), &mn, dummy.Matrix(1, opts), func() *int { y := 1; return &y }(), dummy.Matrix(1, opts), &mn, work.Off((*m)*(*n)+2*mn), &info)

		if iscl == 1 {
			if anrm > bignum {
				golapack.Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &bignum, &anrm, &mn, func() *int { y := 1; return &y }(), work.MatrixOff((*m)*(*n), mn, opts), &mn, &info)
			}
			if anrm < smlnum {
				golapack.Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &smlnum, &anrm, &mn, func() *int { y := 1; return &y }(), work.MatrixOff((*m)*(*n), mn, opts), &mn, &info)
			}
		}

	} else {

		for i = 1; i <= mn; i++ {
			work.Set((*m)*(*n)+i-1, zero)
		}
	}

	//     Compare s and singular values of work
	goblas.Daxpy(mn, -one, s.Off(0, 1), work.Off((*m)*(*n), 1))
	dqrt12Return = goblas.Dasum(mn, work.Off((*m)*(*n), 1)) / (golapack.Dlamch(Epsilon) * float64(max(*m, *n)))
	if nrmsvl != zero {
		dqrt12Return = dqrt12Return / nrmsvl
	}

	return
}
