package lin

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zqlt03 tests ZUNMQL, which computes Q*C, Q'*C, C*Q or C*Q'.
//
// ZQLT03 compares the results of a call to ZUNMQL with the results of
// forming Q explicitly by a call to ZUNGQL and then performing matrix
// multiplication by a call to ZGEMM.
func Zqlt03(m, n, k *int, af, c, cc, q *mat.CMatrix, lda *int, tau, work *mat.CVector, lwork *int, rwork, result *mat.Vector) {
	var side, trans byte
	var rogue complex128
	var cnorm, eps, one, resid, zero float64
	var info, iside, itrans, j, mc, minmn, nc int
	var err error
	_ = err

	iseed := make([]int, 4)

	zero = 0.0
	one = 1.0
	rogue = (-1.0e+10 + (-1.0e+10)*1i)
	srnamt := &gltest.Common.Srnamc.Srnamt

	iseed[0], iseed[1], iseed[2], iseed[3] = 1988, 1989, 1990, 1991

	eps = golapack.Dlamch(Epsilon)
	minmn = min(*m, *n)

	//     Quick return if possible
	if minmn == 0 {
		result.Set(0, zero)
		result.Set(1, zero)
		result.Set(2, zero)
		result.Set(3, zero)
		return
	}

	//     Copy the last k columns of the factorization to the array Q
	golapack.Zlaset('F', m, m, &rogue, &rogue, q, lda)
	if (*k) > 0 && (*m) > (*k) {
		golapack.Zlacpy('F', toPtr((*m)-(*k)), k, af.Off(0, (*n)-(*k)), lda, q.Off(0, (*m)-(*k)), lda)
	}
	if (*k) > 1 {
		golapack.Zlacpy('U', toPtr((*k)-1), toPtr((*k)-1), af.Off((*m)-(*k), (*n)-(*k)+2-1), lda, q.Off((*m)-(*k), (*m)-(*k)+2-1), lda)
	}

	//     Generate the m-by-m matrix Q
	*srnamt = "ZUNGQL"
	golapack.Zungql(m, m, k, q, lda, tau.Off(minmn-(*k)), work, lwork, &info)

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
			*srnamt = "ZUNMQL"
			if (*k) > 0 {
				golapack.Zunmql(side, trans, &mc, &nc, k, af.Off(0, (*n)-(*k)), lda, tau.Off(minmn-(*k)), cc, lda, work, lwork, &info)
			}

			//           Form explicit product and subtract
			if side == 'L' {
				err = goblas.Zgemm(mat.TransByte(trans), NoTrans, mc, nc, mc, complex(-one, 0), q, c, complex(one, 0), cc)
			} else {
				err = goblas.Zgemm(NoTrans, mat.TransByte(trans), mc, nc, nc, complex(-one, 0), c, q, complex(one, 0), cc)
			}

			//           Compute error in the difference
			resid = golapack.Zlange('1', &mc, &nc, cc, lda, rwork)
			result.Set((iside-1)*2+itrans-1, resid/(float64(max(1, *m))*cnorm*eps))

		}
	}
}
