package lin

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/golapack/gltest/matgen"
	"github.com/whipstein/golinalg/mat"
)

// Dqrt15 generates a matrix with full or deficient rank and of various
// norms.
func Dqrt15(scale, rksel, m, n, nrhs *int, a *mat.Matrix, lda *int, b *mat.Matrix, ldb *int, s *mat.Vector, rank *int, norma, normb *float64, iseed *[]int, work *mat.Vector, lwork *int) {
	var bignum, eps, one, smlnum, svmin, temp, two, zero float64
	var info, j, mn int

	dummy := vf(1)

	zero = 0.0
	one = 1.0
	two = 2.0
	svmin = 0.1

	mn = minint(*m, *n)
	if (*lwork) < maxint((*m)+mn, mn*(*nrhs), 2*(*n)+(*m)) {
		gltest.Xerbla([]byte("DQRT15"), 16)
		return
	}

	smlnum = golapack.Dlamch(SafeMinimum)
	bignum = one / smlnum
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
		gltest.Xerbla([]byte("DQRT15"), 2)
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
		golapack.Dlarnv(func() *int { y := 2; return &y }(), iseed, m, work)
		goblas.Dscal(m, toPtrf64(one/goblas.Dnrm2(m, work, toPtr(1))), work, toPtr(1))
		golapack.Dlaset('F', m, rank, &zero, &one, a, lda)
		golapack.Dlarf('L', m, rank, work, func() *int { y := 1; return &y }(), &two, a, lda, work.Off((*m)+1-1))

		//        workspace used: m+mn
		//
		//        Generate consistent rhs in the range space of A
		golapack.Dlarnv(func() *int { y := 2; return &y }(), iseed, toPtr((*rank)*(*nrhs)), work)
		goblas.Dgemm(NoTrans, NoTrans, m, nrhs, rank, &one, a, lda, work.Matrix(*rank, opts), rank, &zero, b, ldb)

		//        work space used: <= mn *nrhs
		//
		//        generate (unscaled) matrix A
		for j = 1; j <= (*rank); j++ {
			goblas.Dscal(m, s.GetPtr(j-1), a.Vector(0, j-1), toPtr(1))
		}
		if (*rank) < (*n) {
			golapack.Dlaset('F', m, toPtr((*n)-(*rank)), &zero, &zero, a.Off(0, (*rank)+1-1), lda)
		}
		matgen.Dlaror('R', 'N', m, n, a, lda, iseed, work, &info)

	} else {
		//        work space used 2*n+m
		//
		//        Generate null matrix and rhs
		for j = 1; j <= mn; j++ {
			s.Set(j-1, zero)
		}
		golapack.Dlaset('F', m, n, &zero, &zero, a, lda)
		golapack.Dlaset('F', m, nrhs, &zero, &zero, b, ldb)

	}

	//     Scale the matrix
	if (*scale) != 1 {
		(*norma) = golapack.Dlange('M', m, n, a, lda, dummy)
		if (*norma) != zero {
			if (*scale) == 2 {
				//              matrix scaled up
				golapack.Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), norma, &bignum, m, n, a, lda, &info)
				golapack.Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), norma, &bignum, &mn, func() *int { y := 1; return &y }(), s.Matrix(mn, opts), &mn, &info)
				golapack.Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), norma, &bignum, m, nrhs, b, ldb, &info)
			} else if (*scale) == 3 {
				//              matrix scaled down
				golapack.Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), norma, &smlnum, m, n, a, lda, &info)
				golapack.Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), norma, &smlnum, &mn, func() *int { y := 1; return &y }(), s.Matrix(mn, opts), &mn, &info)
				golapack.Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), norma, &smlnum, m, nrhs, b, ldb, &info)
			} else {
				gltest.Xerbla([]byte("DQRT15"), 1)
				return
			}
		}
	}

	(*norma) = goblas.Dasum(&mn, s, toPtr(1))
	(*normb) = golapack.Dlange('O', m, nrhs, b, ldb, dummy)
}
