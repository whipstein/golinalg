package lin

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dqlt03 tests DORMQL, which computes Q*C, Q'*C, C*Q or C*Q'.
//
// DQLT03 compares the results of a call to DORMQL with the results of
// forming Q explicitly by a call to DORGQL and then performing matrix
// multiplication by a call to DGEMM.
func Dqlt03(m, n, k *int, af, c, cc, q *mat.Matrix, lda *int, tau, work *mat.Vector, lwork *int, rwork, result *mat.Vector) {
	var side, trans byte
	var cnorm, eps, one, resid, rogue, zero float64
	var info, iside, itrans, j, mc, minmn, nc int
	var err error
	_ = err

	iseed := make([]int, 4)
	srnamt := &gltest.Common.Srnamc.Srnamt

	zero = 0.0
	one = 1.0
	rogue = -1.0e+10

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
	golapack.Dlaset('F', m, m, &rogue, &rogue, q, lda)
	if (*k) > 0 && (*m) > (*k) {
		golapack.Dlacpy('F', toPtr((*m)-(*k)), k, af.Off(0, (*n)-(*k)), lda, q.Off(0, (*m)-(*k)), lda)
	}
	if (*k) > 1 {
		golapack.Dlacpy('U', toPtr((*k)-1), toPtr((*k)-1), af.Off((*m)-(*k), (*n)-(*k)+2-1), lda, q.Off((*m)-(*k), (*m)-(*k)+2-1), lda)
	}

	//     Generate the m-by-m matrix Q
	*srnamt = "DORGQL"
	golapack.Dorgql(m, m, k, q, lda, tau.Off(minmn-(*k)), work, lwork, &info)

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
			*srnamt = "DORMQL"
			if (*k) > 0 {
				golapack.Dormql(side, trans, &mc, &nc, k, af.Off(0, (*n)-(*k)), lda, tau.Off(minmn-(*k)), cc, lda, work, lwork, &info)
			}

			//           Form explicit product and subtract
			if side == 'L' {
				err = goblas.Dgemm(mat.TransByte(trans), mat.NoTrans, mc, nc, mc, -one, q, c, one, cc)
			} else {
				err = goblas.Dgemm(mat.NoTrans, mat.TransByte(trans), mc, nc, nc, -one, c, q, one, cc)
			}

			//           Compute error in the difference
			resid = golapack.Dlange('1', &mc, &nc, cc, lda, rwork)
			result.Set((iside-1)*2+itrans-1, resid/(float64(max(1, *m))*cnorm*eps))

		}
	}
}
