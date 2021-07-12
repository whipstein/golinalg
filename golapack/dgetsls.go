package golapack

import (
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dgetsls solves overdetermined or underdetermined real linear systems
// involving an M-by-N matrix A, using a tall skinny QR or short wide LQ
// factorization of A.  It is assumed that A has full rank.
//
//
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
// 3. If TRANS = 'T' and m >= n:  find the minimum norm solution of
//    an undetermined system A**T * X = B.
//
// 4. If TRANS = 'T' and m < n:  find the least squares solution of
//    an overdetermined system, i.e., solve the least squares problem
//                 minimize || B - A**T * X ||.
//
// Several right hand side vectors b and solution vectors x can be
// handled in a single call; they are stored as the columns of the
// M-by-NRHS right hand side matrix B and the N-by-NRHS solution
// matrix X.
func Dgetsls(trans byte, m, n, nrhs *int, a *mat.Matrix, lda *int, b *mat.Matrix, ldb *int, work *mat.Vector, lwork, info *int) {
	var lquery, tran bool
	var anrm, bignum, bnrm, one, smlnum, zero float64
	var brow, i, iascl, ibscl, info2, j, lw1, lw2, lwm, lwo, maxmn, scllen, tszm, tszo, wsizem, wsizeo int

	tq := vf(5)
	workq := vf(1)

	zero = 0.0
	one = 1.0

	//     Test the input arguments.
	(*info) = 0
	// minmn = min(*m, *n)
	maxmn = max(*m, *n)
	// mnk = max(minmn, *nrhs)
	tran = trans == 'T'

	lquery = ((*lwork) == -1 || (*lwork) == -2)
	if !(trans == 'N' || trans == 'T') {
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
	}

	if (*info) == 0 {
		//     Determine the block size and minimum LWORK
		if (*m) >= (*n) {
			Dgeqr(m, n, a, lda, tq, toPtr(-1), workq, toPtr(-1), &info2)
			tszo = int(tq.Get(0))
			lwo = int(workq.Get(0))
			Dgemqr('L', trans, m, nrhs, n, a, lda, tq, &tszo, b, ldb, workq, toPtr(-1), &info2)
			lwo = max(lwo, int(workq.Get(0)))
			Dgeqr(m, n, a, lda, tq, toPtr(-2), workq, toPtr(-2), &info2)
			tszm = int(tq.Get(0))
			lwm = int(workq.Get(0))
			Dgemqr('L', trans, m, nrhs, n, a, lda, tq, &tszm, b, ldb, workq, toPtr(-1), &info2)
			lwm = max(lwm, int(workq.Get(0)))
			wsizeo = tszo + lwo
			wsizem = tszm + lwm
		} else {
			Dgelq(m, n, a, lda, tq, toPtr(-1), workq, toPtr(-1), &info2)
			tszo = int(tq.Get(0))
			lwo = int(workq.Get(0))
			Dgemlq('L', trans, n, nrhs, m, a, lda, tq, &tszo, b, ldb, workq, toPtr(-1), &info2)
			lwo = max(lwo, int(workq.Get(0)))
			Dgelq(m, n, a, lda, tq, toPtr(-2), workq, toPtr(-2), &info2)
			tszm = int(tq.Get(0))
			lwm = int(workq.Get(0))
			Dgemlq('L', trans, n, nrhs, m, a, lda, tq, &tszo, b, ldb, workq, toPtr(-1), &info2)
			lwm = max(lwm, int(workq.Get(0)))
			wsizeo = tszo + lwo
			wsizem = tszm + lwm
		}

		if ((*lwork) < wsizem) && (!lquery) {
			(*info) = -10
		}

	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("DGETSLS"), -(*info))
		work.Set(0, float64(wsizeo))
		return
	}
	if lquery {
		if (*lwork) == -1 {
			work.Set(0, float64(wsizeo))
		}
		if (*lwork) == -2 {
			work.Set(0, float64(wsizem))
		}
		return
	}
	if (*lwork) < wsizeo {
		lw1 = tszm
		lw2 = lwm
	} else {
		lw1 = tszo
		lw2 = lwo
	}

	//     Quick return if possible
	if min(*m, *n, *nrhs) == 0 {
		Dlaset('F', toPtr(max(*m, *n)), nrhs, &zero, &zero, b, ldb)
		return
	}

	//     Get machine parameters
	smlnum = Dlamch(SafeMinimum) / Dlamch(Precision)
	bignum = one / smlnum
	Dlabad(&smlnum, &bignum)

	//     Scale A, B if max element outside range [SMLNUM,BIGNUM]
	anrm = Dlange('M', m, n, a, lda, work)
	iascl = 0
	if anrm > zero && anrm < smlnum {
		//        Scale matrix norm up to SMLNUM
		Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &anrm, &smlnum, m, n, a, lda, info)
		iascl = 1
	} else if anrm > bignum {
		//        Scale matrix norm down to BIGNUM
		Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &anrm, &bignum, m, n, a, lda, info)
		iascl = 2
	} else if anrm == zero {
		//        Matrix all zero. Return zero solution.
		Dlaset('F', &maxmn, nrhs, &zero, &zero, b, ldb)
		goto label50
	}

	brow = (*m)
	if tran {
		brow = (*n)
	}
	bnrm = Dlange('M', &brow, nrhs, b, ldb, work)
	ibscl = 0
	if bnrm > zero && bnrm < smlnum {
		//        Scale matrix norm up to SMLNUM
		Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &bnrm, &smlnum, &brow, nrhs, b, ldb, info)
		ibscl = 1
	} else if bnrm > bignum {
		//        Scale matrix norm down to BIGNUM
		Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &bnrm, &bignum, &brow, nrhs, b, ldb, info)
		ibscl = 2
	}

	if (*m) >= (*n) {
		//        compute QR factorization of A
		Dgeqr(m, n, a, lda, work.Off(lw2), &lw1, work, &lw2, info)
		if !tran {
			//           Least-Squares Problem min || A * X - B ||
			//
			//           B(1:M,1:NRHS) := Q**T * B(1:M,1:NRHS)
			Dgemqr('L', 'T', m, nrhs, n, a, lda, work.Off(lw2), &lw1, b, ldb, work, &lw2, info)

			//           B(1:N,1:NRHS) := inv(R) * B(1:N,1:NRHS)
			Dtrtrs('U', 'N', 'N', n, nrhs, a, lda, b, ldb, info)
			if (*info) > 0 {
				return
			}
			scllen = (*n)
		} else {
			//           Overdetermined system of equations A**T * X = B
			//
			//           B(1:N,1:NRHS) := inv(R**T) * B(1:N,1:NRHS)
			Dtrtrs('U', 'T', 'N', n, nrhs, a, lda, b, ldb, info)

			if (*info) > 0 {
				return
			}

			//           B(N+1:M,1:NRHS) = ZERO
			for j = 1; j <= (*nrhs); j++ {
				for i = (*n) + 1; i <= (*m); i++ {
					b.Set(i-1, j-1, zero)
				}
			}

			//           B(1:M,1:NRHS) := Q(1:N,:) * B(1:N,1:NRHS)
			Dgemqr('L', 'N', m, nrhs, n, a, lda, work.Off(lw2), &lw1, b, ldb, work, &lw2, info)

			scllen = (*m)

		}

	} else {
		//        Compute LQ factorization of A
		Dgelq(m, n, a, lda, work.Off(lw2), &lw1, work, &lw2, info)

		//        workspace at least M, optimally M*NB.
		if !tran {
			//           underdetermined system of equations A * X = B
			//
			//           B(1:M,1:NRHS) := inv(L) * B(1:M,1:NRHS)
			Dtrtrs('L', 'N', 'N', m, nrhs, a, lda, b, ldb, info)

			if (*info) > 0 {
				return
			}

			//           B(M+1:N,1:NRHS) = 0
			for j = 1; j <= (*nrhs); j++ {
				for i = (*m) + 1; i <= (*n); i++ {
					b.Set(i-1, j-1, zero)
				}
			}

			//           B(1:N,1:NRHS) := Q(1:N,:)**T * B(1:M,1:NRHS)
			Dgemlq('L', 'T', n, nrhs, m, a, lda, work.Off(lw2), &lw1, b, ldb, work, &lw2, info)

			//           workspace at least NRHS, optimally NRHS*NB
			scllen = (*n)

		} else {
			//           overdetermined system min || A**T * X - B ||
			//
			//           B(1:N,1:NRHS) := Q * B(1:N,1:NRHS)
			Dgemlq('L', 'N', n, nrhs, m, a, lda, work.Off(lw2), &lw1, b, ldb, work, &lw2, info)

			//           workspace at least NRHS, optimally NRHS*NB
			//
			//           B(1:M,1:NRHS) := inv(L**T) * B(1:M,1:NRHS)
			Dtrtrs('L', 'T', 'N', m, nrhs, a, lda, b, ldb, info)

			if (*info) > 0 {
				return
			}

			scllen = (*m)

		}

	}

	//     Undo scaling
	if iascl == 1 {
		Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &anrm, &smlnum, &scllen, nrhs, b, ldb, info)
	} else if iascl == 2 {
		Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &anrm, &bignum, &scllen, nrhs, b, ldb, info)
	}
	if ibscl == 1 {
		Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &smlnum, &bnrm, &scllen, nrhs, b, ldb, info)
	} else if ibscl == 2 {
		Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &bignum, &bnrm, &scllen, nrhs, b, ldb, info)
	}

label50:
	;
	work.Set(0, float64(tszo+lwo))
}
