package lin

import (
	"golinalg/goblas"
	"golinalg/golapack"
	"golinalg/golapack/gltest"
	"golinalg/mat"
)

// Dlqt03 tests DORMLQ, which computes Q*C, Q'*C, C*Q or C*Q'.
//
// DLQT03 compares the results of a call to DORMLQ with the results of
// forming Q explicitly by a call to DORGLQ and then performing matrix
// multiplication by a call to DGEMM.
func Dlqt03(m, n, k *int, af, c, cc, q *mat.Matrix, lda *int, tau, work *mat.Vector, lwork *int, rwork, result *mat.Vector) {
	var side, trans byte
	var cnorm, eps, one, resid, rogue float64
	var info, iside, itrans, j, mc, nc int
	iseed := make([]int, 4)
	srnamt := &gltest.Common.Srnamc.Srnamt

	one = 1.0
	rogue = -1.0e+10

	iseed[0], iseed[1], iseed[2], iseed[3] = 1988, 1989, 1990, 1991

	eps = golapack.Dlamch(Epsilon)

	//     Copy the first k rows of the factorization to the array Q
	golapack.Dlaset('F', n, n, &rogue, &rogue, q, lda)
	golapack.Dlacpy('U', k, toPtr((*n)-1), af.Off(0, 1), lda, q.Off(0, 1), lda)

	//     Generate the n-by-n matrix Q
	*srnamt = "DORGLQ"
	golapack.Dorglq(n, n, k, q, lda, tau, work, lwork, &info)

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
			golapack.Dlarnv(func() *int { y := 2; return &y }(), &iseed, &mc, c.Vector(0, j-1))
		}
		cnorm = golapack.Dlange('1', &mc, &nc, c, lda, rwork)
		if cnorm == 0.0 {
			cnorm = one
		}

		for itrans = 1; itrans <= 2; itrans++ {
			if itrans == 1 {
				trans = 'N'
			} else {
				trans = 'T'
			}

			//           Copy C
			golapack.Dlacpy('F', &mc, &nc, c, lda, cc, lda)

			//           Apply Q or Q' to C
			*srnamt = "DORMLQ"
			golapack.Dormlq(side, trans, &mc, &nc, k, af, lda, tau, cc, lda, work, lwork, &info)

			//           Form explicit product and subtract
			if side == 'L' {
				goblas.Dgemm(mat.TransByte(trans), NoTrans, &mc, &nc, &mc, toPtrf64(-one), q, lda, c, lda, &one, cc, lda)
			} else {
				goblas.Dgemm(NoTrans, mat.TransByte(trans), &mc, &nc, &nc, toPtrf64(-one), c, lda, q, lda, &one, cc, lda)
			}

			//           Compute error in the difference
			resid = golapack.Dlange('1', &mc, &nc, cc, lda, rwork)
			result.Set((iside-1)*2+itrans-1, resid/(float64(maxint(1, *n))*cnorm*eps))

		}
	}
}
