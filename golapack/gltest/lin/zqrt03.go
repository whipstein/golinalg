package lin

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zqrt03 tests ZUNMQR, which computes Q*C, Q'*C, C*Q or C*Q'.
//
// ZQRT03 compares the results of a call to ZUNMQR with the results of
// forming Q explicitly by a call to ZUNGQR and then performing matrix
// multiplication by a call to ZGEMM.
func Zqrt03(m, n, k *int, af, c, cc, q *mat.CMatrix, lda *int, tau, work *mat.CVector, lwork *int, rwork, result *mat.Vector) {
	var side, trans byte
	var rogue complex128
	var cnorm, eps, one, resid, zero float64
	var info, iside, itrans, j, mc, nc int
	iseed := make([]int, 4)

	zero = 0.0
	one = 1.0
	rogue = (-1.0e+10 + (-1.0e+10)*1i)

	srnamt := &gltest.Common.Srnamc.Srnamt

	iseed[0], iseed[1], iseed[2], iseed[3] = 1988, 1989, 1990, 1991

	eps = golapack.Dlamch(Epsilon)

	//     Copy the first k columns of the factorization to the array Q
	golapack.Zlaset('F', m, m, &rogue, &rogue, q, lda)
	golapack.Zlacpy('L', toPtr((*m)-1), k, af.Off(1, 0), lda, q.Off(1, 0), lda)

	//     Generate the m-by-m matrix Q
	*srnamt = "ZUNGQR"
	golapack.Zungqr(m, m, k, q, lda, tau, work, lwork, &info)

	for iside = 1; iside <= 2; iside++ {
		if iside == 1 {
			side = 'L'
			mc = (*m)
			nc = (*n)
		} else {
			side = 'R'
			mc = (*n)
			nc = (*m)
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
			*srnamt = "ZUNMQR"
			golapack.Zunmqr(side, trans, &mc, &nc, k, af, lda, tau, cc, lda, work, lwork, &info)

			//           Form explicit product and subtract
			if side == 'L' {
				goblas.Zgemm(mat.TransByte(trans), NoTrans, &mc, &nc, &mc, toPtrc128(complex(-one, 0)), q, lda, c, lda, toPtrc128(complex(one, 0)), cc, lda)
			} else {
				goblas.Zgemm(NoTrans, mat.TransByte(trans), &mc, &nc, &nc, toPtrc128(complex(-one, 0)), c, lda, q, lda, toPtrc128(complex(one, 0)), cc, lda)
			}

			//           Compute error in the difference
			resid = golapack.Zlange('1', &mc, &nc, cc, lda, rwork)
			result.Set((iside-1)*2+itrans-1, resid/(float64(maxint(1, *m))*cnorm*eps))

		}
	}
}
