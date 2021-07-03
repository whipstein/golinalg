package lin

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/golapack/gltest/matgen"
	"github.com/whipstein/golinalg/mat"
)

// Zqrt15 generates a matrix with full or deficient rank and of various
// norms.
func Zqrt15(scale, rksel, m, n, nrhs *int, a *mat.CMatrix, lda *int, b *mat.CMatrix, ldb *int, s *mat.Vector, rank *int, norma, normb *float64, iseed *[]int, work *mat.CVector, lwork *int) {
	var cone, czero complex128
	var bignum, eps, one, smlnum, svmin, temp, two, zero float64
	var info, j, mn int
	var err error
	_ = err

	dummy := vf(1)

	zero = 0.0
	one = 1.0
	two = 2.0
	svmin = 0.1
	czero = (0.0 + 0.0*1i)
	cone = (1.0 + 0.0*1i)

	mn = minint(*m, *n)
	if (*lwork) < maxint((*m)+mn, mn*(*nrhs), 2*(*n)+(*m)) {
		gltest.Xerbla([]byte("ZQRT15"), 16)
		return
	}

	smlnum = golapack.Dlamch(SafeMinimum)
	bignum = one / smlnum
	golapack.Dlabad(&smlnum, &bignum)
	eps = golapack.Dlamch(Epsilon)
	smlnum = (smlnum / eps) / eps
	bignum = one / smlnum

	//     Determine rank and (unscaled) singular values
	if (*rksel) == 1 {
		(*rank) = mn
	} else if (*rksel) == 2 {
		(*rank) = (3 * mn) / 4
		for j = (*rank) + 1; j <= mn; j++ {
			s.Set(j-1, zero)
		}
	} else {
		gltest.Xerbla([]byte("ZQRT15"), 2)
	}

	if (*rank) > 0 {
		//        Nontrivial case
		s.Set(0, one)
		for j = 2; j <= (*rank); j++ {
		label20:
			;
			temp = matgen.Dlarnd(func() *int { y := 1; return &y }(), iseed)
			if temp > svmin {
				s.Set(j-1, math.Abs(temp))
			} else {
				goto label20
			}
		}
		Dlaord('D', rank, s, func() *int { y := 1; return &y }())

		//        Generate 'rank' columns of a random orthogonal matrix in A
		golapack.Zlarnv(func() *int { y := 2; return &y }(), iseed, m, work)
		goblas.Zdscal(*m, one/goblas.Dznrm2(*m, work, 1), work, 1)
		golapack.Zlaset('F', m, rank, &czero, &cone, a, lda)
		golapack.Zlarf('L', m, rank, work, func() *int { y := 1; return &y }(), toPtrc128(complex(two, 0)), a, lda, work.Off((*m)+1-1))

		//        workspace used: m+mn
		//
		//        Generate consistent rhs in the range space of A
		golapack.Zlarnv(func() *int { y := 2; return &y }(), iseed, toPtr((*rank)*(*nrhs)), work)
		err = goblas.Zgemm(NoTrans, NoTrans, *m, *nrhs, *rank, cone, a, *lda, work.CMatrix(*rank, opts), *rank, czero, b, *ldb)

		//        work space used: <= mn *nrhs
		//
		//        generate (unscaled) matrix A
		for j = 1; j <= (*rank); j++ {
			goblas.Zdscal(*m, s.Get(j-1), a.CVector(0, j-1), 1)
		}
		if (*rank) < (*n) {
			golapack.Zlaset('F', m, toPtr((*n)-(*rank)), &czero, &czero, a.Off(0, (*rank)+1-1), lda)
		}
		matgen.Zlaror('R', 'N', m, n, a, lda, iseed, work, &info)

	} else {
		//        work space used 2*n+m
		//
		//        Generate null matrix and rhs
		for j = 1; j <= mn; j++ {
			s.Set(j-1, zero)
		}
		golapack.Zlaset('F', m, n, &czero, &czero, a, lda)
		golapack.Zlaset('F', m, nrhs, &czero, &czero, b, ldb)

	}

	//     Scale the matrix
	if (*scale) != 1 {
		(*norma) = golapack.Zlange('M', m, n, a, lda, dummy)
		if (*norma) != zero {
			if (*scale) == 2 {
				//              matrix scaled up
				golapack.Zlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), norma, &bignum, m, n, a, lda, &info)
				golapack.Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), norma, &bignum, &mn, func() *int { y := 1; return &y }(), s.Matrix(mn, opts), &mn, &info)
				golapack.Zlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), norma, &bignum, m, nrhs, b, ldb, &info)
			} else if (*scale) == 3 {
				//              matrix scaled down
				golapack.Zlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), norma, &smlnum, m, n, a, lda, &info)
				golapack.Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), norma, &smlnum, &mn, func() *int { y := 1; return &y }(), s.Matrix(mn, opts), &mn, &info)
				golapack.Zlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), norma, &smlnum, m, nrhs, b, ldb, &info)
			} else {
				gltest.Xerbla([]byte("ZQRT15"), 1)
				return
			}
		}
	}

	(*norma) = goblas.Dasum(mn, s, 1)
	(*normb) = golapack.Zlange('O', m, nrhs, b, ldb, dummy)
}
