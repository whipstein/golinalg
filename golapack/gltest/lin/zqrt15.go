package lin

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/golapack/gltest/matgen"
	"github.com/whipstein/golinalg/mat"
)

// zqrt15 generates a matrix with full or deficient rank and of various
// norms.
func zqrt15(scale, rksel, m, n, nrhs int, a, b *mat.CMatrix, s *mat.Vector, iseed *[]int, work *mat.CVector, lwork int) (rank int, norma, normb float64, err error) {
	var cone, czero complex128
	var bignum, eps, one, smlnum, svmin, temp, two, zero float64
	var j, mn int

	dummy := vf(1)

	zero = 0.0
	one = 1.0
	two = 2.0
	svmin = 0.1
	czero = (0.0 + 0.0*1i)
	cone = (1.0 + 0.0*1i)

	mn = min(m, n)
	if lwork < max(m+mn, mn*nrhs, 2*n+m) {
		err = fmt.Errorf("lwork < max(m+mn, mn*nrhs, 2*n+m); lwork=%v, m=%v, n=%v, nrhs=%v", lwork, m, n, nrhs)
		gltest.Xerbla2("zqrt15", err)
		return
	}

	smlnum = golapack.Dlamch(SafeMinimum)
	bignum = one / smlnum
	smlnum, bignum = golapack.Dlabad(smlnum, bignum)
	eps = golapack.Dlamch(Epsilon)
	smlnum = (smlnum / eps) / eps
	bignum = one / smlnum

	//     Determine rank and (unscaled) singular values
	if rksel == 1 {
		rank = mn
	} else if rksel == 2 {
		rank = (3 * mn) / 4
		for j = rank + 1; j <= mn; j++ {
			s.Set(j-1, zero)
		}
	} else {
		err = fmt.Errorf("rksel=%v", rksel)
		gltest.Xerbla2("zqrt15", err)
	}

	if rank > 0 {
		//        Nontrivial case
		s.Set(0, one)
		for j = 2; j <= rank; j++ {
		label20:
			;
			temp = matgen.Dlarnd(1, iseed)
			if temp > svmin {
				s.Set(j-1, math.Abs(temp))
			} else {
				goto label20
			}
		}
		dlaord('D', rank, s, 1)

		//        Generate 'rank' columns of a random orthogonal matrix in A
		golapack.Zlarnv(2, iseed, m, work)
		work.Dscal(m, one/work.Nrm2(m, 1), 1)
		golapack.Zlaset(Full, m, rank, czero, cone, a)
		golapack.Zlarf(Left, m, rank, work, 1, complex(two, 0), a, work.Off(m))

		//        workspace used: m+mn
		//
		//        Generate consistent rhs in the range space of A
		golapack.Zlarnv(2, iseed, rank*nrhs, work)
		if err = b.Gemm(NoTrans, NoTrans, m, nrhs, rank, cone, a, work.CMatrix(rank, opts), czero); err != nil {
			panic(err)
		}

		//        work space used: <= mn *nrhs
		//
		//        generate (unscaled) matrix A
		for j = 1; j <= rank; j++ {
			a.Off(0, j-1).CVector().Dscal(m, s.Get(j-1), 1)
		}
		if rank < n {
			golapack.Zlaset(Full, m, n-rank, czero, czero, a.Off(0, rank))
		}
		if err = matgen.Zlaror('R', 'N', m, n, a, iseed, work); err != nil {
			panic(err)
		}

	} else {
		//        work space used 2*n+m
		//
		//        Generate null matrix and rhs
		for j = 1; j <= mn; j++ {
			s.Set(j-1, zero)
		}
		golapack.Zlaset(Full, m, n, czero, czero, a)
		golapack.Zlaset(Full, m, nrhs, czero, czero, b)

	}

	//     Scale the matrix
	if scale != 1 {
		norma = golapack.Zlange('M', m, n, a, dummy)
		if norma != zero {
			if scale == 2 {
				//              matrix scaled up
				if err = golapack.Zlascl('G', 0, 0, norma, bignum, m, n, a); err != nil {
					panic(err)
				}
				if err = golapack.Dlascl('G', 0, 0, norma, bignum, mn, 1, s.Matrix(mn, opts)); err != nil {
					panic(err)
				}
				if err = golapack.Zlascl('G', 0, 0, norma, bignum, m, nrhs, b); err != nil {
					panic(err)
				}
			} else if scale == 3 {
				//              matrix scaled down
				if err = golapack.Zlascl('G', 0, 0, norma, smlnum, m, n, a); err != nil {
					panic(err)
				}
				if err = golapack.Dlascl('G', 0, 0, norma, smlnum, mn, 1, s.Matrix(mn, opts)); err != nil {
					panic(err)
				}
				if err = golapack.Zlascl('G', 0, 0, norma, smlnum, m, nrhs, b); err != nil {
					panic(err)
				}
			} else {
				err = fmt.Errorf("scale=%v", scale)
				gltest.Xerbla2("zqrt15", err)
				return
			}
		}
	}

	norma = s.Asum(mn, 1)
	normb = golapack.Zlange('O', m, nrhs, b, dummy)

	return
}
