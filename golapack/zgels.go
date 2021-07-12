package golapack

import (
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zgels solves overdetermined or underdetermined complex linear systems
// involving an M-by-N matrix A, or its conjugate-transpose, using a QR
// or LQ factorization of A.  It is assumed that A has full rank.
//
// The following options are provided:
//
// 1. If TRANS = 'N' and m >= n:  find the least squares solution of
//    an overdetermined system, i.e., solve the least squares problem
//                 minimize || B - A*X ||.
//
// 2. If TRANS = 'N' and m < n:  find the minimum norm solution of
//    an underdetermined system A * X = B.
//
// 3. If TRANS = 'C' and m >= n:  find the minimum norm solution of
//    an underdetermined system A**H * X = B.
//
// 4. If TRANS = 'C' and m < n:  find the least squares solution of
//    an overdetermined system, i.e., solve the least squares problem
//                 minimize || B - A**H * X ||.
//
// Several right hand side vectors b and solution vectors x can be
// handled in a single call; they are stored as the columns of the
// M-by-NRHS right hand side matrix B and the N-by-NRHS solution
// matrix X.
func Zgels(trans byte, m, n, nrhs *int, a *mat.CMatrix, lda *int, b *mat.CMatrix, ldb *int, work *mat.CVector, lwork, info *int) {
	var lquery, tpsd bool
	var czero complex128
	var anrm, bignum, bnrm, one, smlnum, zero float64
	var brow, i, iascl, ibscl, j, mn, nb, scllen, wsize int

	rwork := vf(1)

	zero = 0.0
	one = 1.0
	czero = (0.0 + 0.0*1i)

	//     Test the input arguments.
	(*info) = 0
	mn = min(*m, *n)
	lquery = ((*lwork) == -1)
	if !(trans == 'N' || trans == 'C') {
		(*info) = -1
	} else if (*m) < 0 {
		(*info) = -2
	} else if (*n) < 0 {
		(*info) = -3
	} else if (*nrhs) < 0 {
		(*info) = -4
	} else if (*lda) < max(1, *m) {
		(*info) = -6
	} else if (*ldb) < max(1, *m, *n) {
		(*info) = -8
	} else if (*lwork) < max(1, mn+max(mn, *nrhs)) && !lquery {
		(*info) = -10
	}

	//     Figure out optimal block size
	if (*info) == 0 || (*info) == -10 {

		tpsd = true
		if trans == 'N' {
			tpsd = false
		}

		if (*m) >= (*n) {
			nb = Ilaenv(func() *int { y := 1; return &y }(), []byte("ZGEQRF"), []byte{' '}, m, n, toPtr(-1), toPtr(-1))
			if tpsd {
				nb = max(nb, Ilaenv(func() *int { y := 1; return &y }(), []byte("ZUNMQR"), []byte("LN"), m, nrhs, n, toPtr(-1)))
			} else {
				nb = max(nb, Ilaenv(func() *int { y := 1; return &y }(), []byte("ZUNMQR"), []byte("LC"), m, nrhs, n, toPtr(-1)))
			}
		} else {
			nb = Ilaenv(func() *int { y := 1; return &y }(), []byte("ZGELQF"), []byte{' '}, m, n, toPtr(-1), toPtr(-1))
			if tpsd {
				nb = max(nb, Ilaenv(func() *int { y := 1; return &y }(), []byte("ZUNMLQ"), []byte("LC"), n, nrhs, m, toPtr(-1)))
			} else {
				nb = max(nb, Ilaenv(func() *int { y := 1; return &y }(), []byte("ZUNMLQ"), []byte("LN"), n, nrhs, m, toPtr(-1)))
			}
		}

		wsize = max(1, mn+max(mn, *nrhs)*nb)
		work.SetRe(0, float64(wsize))

	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("ZGELS "), -(*info))
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if min(*m, *n, *nrhs) == 0 {
		Zlaset('F', toPtr(max(*m, *n)), nrhs, &czero, &czero, b, ldb)
		return
	}

	//     Get machine parameters
	smlnum = Dlamch(SafeMinimum) / Dlamch(Precision)
	bignum = one / smlnum
	Dlabad(&smlnum, &bignum)

	//     Scale A, B if max element outside range [SMLNUM,BIGNUM]
	//
	anrm = Zlange('M', m, n, a, lda, rwork)
	iascl = 0
	if anrm > zero && anrm < smlnum {
		//        Scale matrix norm up to SMLNUM
		Zlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &anrm, &smlnum, m, n, a, lda, info)
		iascl = 1
	} else if anrm > bignum {
		//        Scale matrix norm down to BIGNUM
		Zlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &anrm, &bignum, m, n, a, lda, info)
		iascl = 2
	} else if anrm == zero {
		//        Matrix all zero. Return zero solution.
		Zlaset('F', toPtr(max(*m, *n)), nrhs, &czero, &czero, b, ldb)
		goto label50
	}
	//
	brow = (*m)
	if tpsd {
		brow = (*n)
	}
	bnrm = Zlange('M', &brow, nrhs, b, ldb, rwork)
	ibscl = 0
	if bnrm > zero && bnrm < smlnum {
		//        Scale matrix norm up to SMLNUM
		Zlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &bnrm, &smlnum, &brow, nrhs, b, ldb, info)
		ibscl = 1
	} else if bnrm > bignum {
		//        Scale matrix norm down to BIGNUM
		Zlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &bnrm, &bignum, &brow, nrhs, b, ldb, info)
		ibscl = 2
	}

	if (*m) >= (*n) {
		//        compute QR factorization of A
		Zgeqrf(m, n, a, lda, work.Off(0), work.Off(mn), toPtr((*lwork)-mn), info)

		//        workspace at least N, optimally N*NB
		if !tpsd {
			//           Least-Squares Problem min || A * X - B ||
			//
			//           B(1:M,1:NRHS) := Q**H * B(1:M,1:NRHS)
			Zunmqr('L', 'C', m, nrhs, n, a, lda, work.Off(0), b, ldb, work.Off(mn), toPtr((*lwork)-mn), info)

			//           workspace at least NRHS, optimally NRHS*NB
			//
			//           B(1:N,1:NRHS) := inv(R) * B(1:N,1:NRHS)
			Ztrtrs('U', 'N', 'N', n, nrhs, a, lda, b, ldb, info)

			if (*info) > 0 {
				return
			}

			scllen = (*n)

		} else {
			//           Underdetermined system of equations A**T * X = B
			//
			//           B(1:N,1:NRHS) := inv(R**H) * B(1:N,1:NRHS)
			Ztrtrs('U', 'C', 'N', n, nrhs, a, lda, b, ldb, info)

			if (*info) > 0 {
				return
			}

			//           B(N+1:M,1:NRHS) = ZERO
			for j = 1; j <= (*nrhs); j++ {
				for i = (*n) + 1; i <= (*m); i++ {
					b.Set(i-1, j-1, czero)
				}
			}

			//           B(1:M,1:NRHS) := Q(1:N,:) * B(1:N,1:NRHS)
			Zunmqr('L', 'N', m, nrhs, n, a, lda, work.Off(0), b, ldb, work.Off(mn), toPtr((*lwork)-mn), info)

			//           workspace at least NRHS, optimally NRHS*NB
			scllen = (*m)

		}

	} else {
		//        Compute LQ factorization of A
		Zgelqf(m, n, a, lda, work.Off(0), work.Off(mn), toPtr((*lwork)-mn), info)

		//        workspace at least M, optimally M*NB.
		if !tpsd {
			//           underdetermined system of equations A * X = B
			//
			//           B(1:M,1:NRHS) := inv(L) * B(1:M,1:NRHS)
			Ztrtrs('L', 'N', 'N', m, nrhs, a, lda, b, ldb, info)

			if (*info) > 0 {
				return
			}

			//           B(M+1:N,1:NRHS) = 0
			for j = 1; j <= (*nrhs); j++ {
				for i = (*m) + 1; i <= (*n); i++ {
					b.Set(i-1, j-1, czero)
				}
			}

			//           B(1:N,1:NRHS) := Q(1:N,:)**H * B(1:M,1:NRHS)
			Zunmlq('L', 'C', n, nrhs, m, a, lda, work.Off(0), b, ldb, work.Off(mn), toPtr((*lwork)-mn), info)

			//           workspace at least NRHS, optimally NRHS*NB
			scllen = (*n)

		} else {
			//           overdetermined system min || A**H * X - B ||
			//
			//           B(1:N,1:NRHS) := Q * B(1:N,1:NRHS)
			Zunmlq('L', 'N', n, nrhs, m, a, lda, work.Off(0), b, ldb, work.Off(mn), toPtr((*lwork)-mn), info)

			//           workspace at least NRHS, optimally NRHS*NB
			//
			//           B(1:M,1:NRHS) := inv(L**H) * B(1:M,1:NRHS)
			Ztrtrs('L', 'C', 'N', m, nrhs, a, lda, b, ldb, info)

			if (*info) > 0 {
				return
			}

			scllen = (*m)

		}

	}

	//     Undo scaling
	if iascl == 1 {
		Zlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &anrm, &smlnum, &scllen, nrhs, b, ldb, info)
	} else if iascl == 2 {
		Zlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &anrm, &bignum, &scllen, nrhs, b, ldb, info)
	}
	if ibscl == 1 {
		Zlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &smlnum, &bnrm, &scllen, nrhs, b, ldb, info)
	} else if ibscl == 2 {
		Zlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &bignum, &bnrm, &scllen, nrhs, b, ldb, info)
	}

label50:
	;
	work.SetRe(0, float64(wsize))
}
