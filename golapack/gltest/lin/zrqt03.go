package lin

import (
	"golinalg/goblas"
	"golinalg/golapack"
	"golinalg/golapack/gltest"
	"golinalg/mat"
)

// Zrqt03 tests ZUNMRQ, which computes Q*C, Q'*C, C*Q or C*Q'.
//
// ZRQT03 compares the results of a call to ZUNMRQ with the results of
// forming Q explicitly by a call to ZUNGRQ and then performing matrix
// multiplication by a call to ZGEMM.
func Zrqt03(m, n, k *int, af, c, cc, q *mat.CMatrix, lda *int, tau, work *mat.CVector, lwork *int, rwork, result *mat.Vector) {
	var side, trans byte
	var rogue complex128
	var cnorm, eps, one, resid, zero float64
	var info, iside, itrans, j, mc, minmn, nc int
	iseed := make([]int, 4)

	zero = 0.0
	one = 1.0
	rogue = (-1.0e+10 + (-1.0e+10)*1i)
	srnamt := &gltest.Common.Srnamc.Srnamt

	iseed[0], iseed[1], iseed[2], iseed[3] = 1988, 1989, 1990, 1991

	eps = golapack.Dlamch(Epsilon)
	minmn = minint(*m, *n)

	//     Quick return if possible
	if minmn == 0 {
		result.Set(0, zero)
		result.Set(1, zero)
		result.Set(2, zero)
		result.Set(3, zero)
		return
	}

	//     Copy the last k rows of the factorization to the array Q
	golapack.Zlaset('F', n, n, &rogue, &rogue, q, lda)
	if (*k) > 0 && (*n) > (*k) {
		golapack.Zlacpy('F', k, toPtr((*n)-(*k)), af.Off((*m)-(*k)+1-1, 0), lda, q.Off((*n)-(*k)+1-1, 0), lda)
	}
	if (*k) > 1 {
		golapack.Zlacpy('L', toPtr((*k)-1), toPtr((*k)-1), af.Off((*m)-(*k)+2-1, (*n)-(*k)+1-1), lda, q.Off((*n)-(*k)+2-1, (*n)-(*k)+1-1), lda)
	}
	//
	//     Generate the n-by-n matrix Q
	//
	*srnamt = "ZUNGRQ"
	golapack.Zungrq(n, n, k, q, lda, tau.Off(minmn-(*k)+1-1), work, lwork, &info)
	//
	for iside = 1; iside <= 2; iside++ {
		if iside == 1 {
			side = 'L'
			mc = (*n)
			nc = (*m)
		} else {
			side = 'R'
			mc = (*m)
			nc = (*n)
		}

		//        Generate MC by NC matrix C
		for j = 1; j <= nc; j++ {
			golapack.Zlarnv(func() *int { y := 2; return &y }(), &iseed, &mc, c.CVector(0, j-1))
		}
		cnorm = golapack.Zlange('1', &mc, &nc, c, lda, rwork)
		if cnorm == zero {
			cnorm = one
		}

		for itrans = 1; itrans <= 2; itrans++ {
			if itrans == 1 {
				trans = 'N'
			} else {
				trans = 'C'
			}

			//           Copy C
			golapack.Zlacpy('F', &mc, &nc, c, lda, cc, lda)

			//           Apply Q or Q' to C
			*srnamt = "ZUNMRQ"
			if (*k) > 0 {
				golapack.Zunmrq(side, trans, &mc, &nc, k, af.Off((*m)-(*k)+1-1, 0), lda, tau.Off(minmn-(*k)+1-1), cc, lda, work, lwork, &info)
			}

			//           Form explicit product and subtract
			if side == 'L' {
				goblas.Zgemm(mat.TransByte(trans), NoTrans, &mc, &nc, &mc, toPtrc128(complex(-one, 0)), q, lda, c, lda, toPtrc128(complex(one, 0)), cc, lda)
			} else {
				goblas.Zgemm(NoTrans, mat.TransByte(trans), &mc, &nc, &nc, toPtrc128(complex(-one, 0)), c, lda, q, lda, toPtrc128(complex(one, 0)), cc, lda)
			}

			//           Compute error in the difference
			resid = golapack.Zlange('1', &mc, &nc, cc, lda, rwork)
			result.Set((iside-1)*2+itrans-1, resid/(float64(maxint(1, *n))*cnorm*eps))

		}
	}
}