package golapack

import (
	"math/cmplx"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zgghd3 reduces a pair of complex matrices (A,B) to generalized upper
// Hessenberg form using unitary transformations, where A is a
// general matrix and B is upper triangular.  The form of the
// generalized eigenvalue problem is
//    A*x = lambda*B*x,
// and B is typically made upper triangular by computing its QR
// factorization and moving the unitary matrix Q to the left side
// of the equation.
//
// This subroutine simultaneously reduces A to a Hessenberg matrix H:
//    Q**H*A*Z = H
// and transforms B to another upper triangular matrix T:
//    Q**H*B*Z = T
// in order to reduce the problem to its standard form
//    H*y = lambda*T*y
// where y = Z**H*x.
//
// The unitary matrices Q and Z are determined as products of Givens
// rotations.  They may either be formed explicitly, or they may be
// postmultiplied into input matrices Q1 and Z1, so that
//      Q1 * A * Z1**H = (Q1*Q) * H * (Z1*Z)**H
//      Q1 * B * Z1**H = (Q1*Q) * T * (Z1*Z)**H
// If Q1 is the unitary matrix from the QR factorization of B in the
// original equation A*x = lambda*B*x, then ZGGHD3 reduces the original
// problem to generalized Hessenberg form.
//
// This is a blocked variant of CGGHRD, using matrix-matrix
// multiplications for parts of the computation to enhance performance.
func Zgghd3(compq, compz byte, n, ilo, ihi *int, a *mat.CMatrix, lda *int, b *mat.CMatrix, ldb *int, q *mat.CMatrix, ldq *int, z *mat.CMatrix, ldz *int, work *mat.CVector, lwork, info *int) {
	var blk22, initq, initz, lquery, wantq, wantz bool
	var compq2, compz2 byte
	var c1, c2, cone, ctemp, czero, s, s1, s2, temp, temp1, temp2, temp3 complex128
	var c float64
	var cola, i, ierr, j, j0, jcol, jj, jrow, k, kacc22, len, lwkopt, n2nb, nb, nblst, nbmin, nh, nnb, nx, ppw, ppwo, pw, top, topq int
	var err error
	_ = err

	cone = (1.0 + 0.0*1i)
	czero = (0.0 + 0.0*1i)

	//     Decode and test the input parameters.
	(*info) = 0
	nb = Ilaenv(func() *int { y := 1; return &y }(), []byte("ZGGHD3"), []byte{' '}, n, ilo, ihi, toPtr(-1))
	lwkopt = maxint(6*(*n)*nb, 1)
	work.SetRe(0, float64(lwkopt))
	initq = compq == 'I'
	wantq = initq || compq == 'V'
	initz = compz == 'I'
	wantz = initz || compz == 'V'
	lquery = ((*lwork) == -1)

	if compq != 'N' && !wantq {
		(*info) = -1
	} else if compz != 'N' && !wantz {
		(*info) = -2
	} else if (*n) < 0 {
		(*info) = -3
	} else if (*ilo) < 1 {
		(*info) = -4
	} else if (*ihi) > (*n) || (*ihi) < (*ilo)-1 {
		(*info) = -5
	} else if (*lda) < maxint(1, *n) {
		(*info) = -7
	} else if (*ldb) < maxint(1, *n) {
		(*info) = -9
	} else if (wantq && (*ldq) < (*n)) || (*ldq) < 1 {
		(*info) = -11
	} else if (wantz && (*ldz) < (*n)) || (*ldz) < 1 {
		(*info) = -13
	} else if (*lwork) < 1 && !lquery {
		(*info) = -15
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZGGHD3"), -(*info))
		return
	} else if lquery {
		return
	}

	//     Initialize Q and Z if desired.
	if initq {
		Zlaset('A', n, n, &czero, &cone, q, ldq)
	}
	if initz {
		Zlaset('A', n, n, &czero, &cone, z, ldz)
	}

	//     Zero out lower triangle of B.
	if (*n) > 1 {
		Zlaset('L', toPtr((*n)-1), toPtr((*n)-1), &czero, &czero, b.Off(1, 0), ldb)
	}

	//     Quick return if possible
	nh = (*ihi) - (*ilo) + 1
	if nh <= 1 {
		work.Set(0, cone)
		return
	}

	//     Determine the blocksize.
	nbmin = Ilaenv(func() *int { y := 2; return &y }(), []byte("ZGGHD3"), []byte{' '}, n, ilo, ihi, toPtr(-1))
	if nb > 1 && nb < nh {
		//        Determine when to use unblocked instead of blocked code.
		nx = maxint(nb, Ilaenv(func() *int { y := 3; return &y }(), []byte("ZGGHD3"), []byte{' '}, n, ilo, ihi, toPtr(-1)))
		if nx < nh {
			//           Determine if workspace is large enough for blocked code.
			if (*lwork) < lwkopt {
				//              Not enough workspace to use optimal NB:  determine the
				//              minimum value of NB, and reduce NB or force use of
				//              unblocked code.
				nbmin = maxint(2, Ilaenv(func() *int { y := 2; return &y }(), []byte("ZGGHD3"), []byte{' '}, n, ilo, ihi, toPtr(-1)))
				if (*lwork) >= 6*(*n)*nbmin {
					nb = (*lwork) / (6 * (*n))
				} else {
					nb = 1
				}
			}
		}
	}

	if nb < nbmin || nb >= nh {
		//        Use unblocked code below
		jcol = (*ilo)

	} else {
		//        Use blocked code
		kacc22 = Ilaenv(func() *int { y := 16; return &y }(), []byte("ZGGHD3"), []byte{' '}, n, ilo, ihi, toPtr(-1))
		blk22 = kacc22 == 2
		for jcol = (*ilo); jcol <= (*ihi)-2; jcol += nb {
			nnb = minint(nb, (*ihi)-jcol-1)

			//           Initialize small unitary factors that will hold the
			//           accumulated Givens rotations in workspace.
			//           N2NB   denotes the number of 2*NNB-by-2*NNB factors
			//           NBLST  denotes the (possibly smaller) order of the last
			//                  factor.
			n2nb = ((*ihi)-jcol-1)/nnb - 1
			nblst = (*ihi) - jcol - n2nb*nnb
			Zlaset('A', &nblst, &nblst, &czero, &cone, work.CMatrix(nblst, opts), &nblst)
			pw = nblst*nblst + 1
			for i = 1; i <= n2nb; i++ {
				Zlaset('A', toPtr(2*nnb), toPtr(2*nnb), &czero, &cone, work.CMatrixOff(pw-1, 2*nnb, opts), toPtr(2*nnb))
				pw = pw + 4*nnb*nnb
			}

			//           Reduce columns JCOL:JCOL+NNB-1 of A to Hessenberg form.
			for j = jcol; j <= jcol+nnb-1; j++ {
				//              Reduce Jth column of A. Store cosines and sines in Jth
				//              column of A and B, respectively.
				for i = (*ihi); i >= j+2; i-- {
					temp = a.Get(i-1-1, j-1)
					Zlartg(&temp, a.GetPtr(i-1, j-1), &c, &s, a.GetPtr(i-1-1, j-1))
					a.SetRe(i-1, j-1, c)
					b.Set(i-1, j-1, s)
				}

				//              Accumulate Givens rotations into workspace array.
				ppw = (nblst+1)*(nblst-2) - j + jcol + 1
				len = 2 + j - jcol
				jrow = j + n2nb*nnb + 2
				for i = (*ihi); i >= jrow; i-- {
					ctemp = a.Get(i-1, j-1)
					s = b.Get(i-1, j-1)
					for jj = ppw; jj <= ppw+len-1; jj++ {
						temp = work.Get(jj + nblst - 1)
						work.Set(jj+nblst-1, ctemp*temp-s*work.Get(jj-1))
						work.Set(jj-1, cmplx.Conj(s)*temp+ctemp*work.Get(jj-1))
					}
					len = len + 1
					ppw = ppw - nblst - 1
				}

				ppwo = nblst*nblst + (nnb+j-jcol-1)*2*nnb + nnb
				j0 = jrow - nnb
				for jrow = j0; jrow >= j+2; jrow -= nnb {
					ppw = ppwo
					len = 2 + j - jcol
					for i = jrow + nnb - 1; i >= jrow; i-- {
						ctemp = a.Get(i-1, j-1)
						s = b.Get(i-1, j-1)
						for jj = ppw; jj <= ppw+len-1; jj++ {
							temp = work.Get(jj + 2*nnb - 1)
							work.Set(jj+2*nnb-1, ctemp*temp-s*work.Get(jj-1))
							work.Set(jj-1, cmplx.Conj(s)*temp+ctemp*work.Get(jj-1))
						}
						len = len + 1
						ppw = ppw - 2*nnb - 1
					}
					ppwo = ppwo + 4*nnb*nnb
				}

				//              TOP denotes the number of top rows in A and B that will
				//              not be updated during the next steps.
				if jcol <= 2 {
					top = 0
				} else {
					top = jcol
				}

				//              Propagate transformations through B and replace stored
				//              left sines/cosines by right sines/cosines.
				for jj = (*n); jj >= j+1; jj-- {
					//                 Update JJth column of B.
					for i = minint(jj+1, *ihi); i >= j+2; i-- {
						ctemp = a.Get(i-1, j-1)
						s = b.Get(i-1, j-1)
						temp = b.Get(i-1, jj-1)
						b.Set(i-1, jj-1, ctemp*temp-cmplx.Conj(s)*b.Get(i-1-1, jj-1))
						b.Set(i-1-1, jj-1, s*temp+ctemp*b.Get(i-1-1, jj-1))
					}

					//                 Annihilate B( JJ+1, JJ ).
					if jj < (*ihi) {
						temp = b.Get(jj+1-1, jj+1-1)
						Zlartg(&temp, b.GetPtr(jj+1-1, jj-1), &c, &s, b.GetPtr(jj+1-1, jj+1-1))
						b.Set(jj+1-1, jj-1, czero)
						Zrot(toPtr(jj-top), b.CVector(top+1-1, jj+1-1), func() *int { y := 1; return &y }(), b.CVector(top+1-1, jj-1), func() *int { y := 1; return &y }(), &c, &s)
						a.SetRe(jj+1-1, j-1, c)
						b.Set(jj+1-1, j-1, -cmplx.Conj(s))
					}
				}

				//              Update A by transformations from right.
				jj = (((*ihi) - j - 1) % 3)
				for i = (*ihi) - j - 3; i >= jj+1; i -= 3 {
					ctemp = a.Get(j+1+i-1, j-1)
					s = -b.Get(j+1+i-1, j-1)
					c1 = a.Get(j+2+i-1, j-1)
					s1 = -b.Get(j+2+i-1, j-1)
					c2 = a.Get(j+3+i-1, j-1)
					s2 = -b.Get(j+3+i-1, j-1)
					//
					for k = top + 1; k <= (*ihi); k++ {
						temp = a.Get(k-1, j+i-1)
						temp1 = a.Get(k-1, j+i+1-1)
						temp2 = a.Get(k-1, j+i+2-1)
						temp3 = a.Get(k-1, j+i+3-1)
						a.Set(k-1, j+i+3-1, c2*temp3+cmplx.Conj(s2)*temp2)
						temp2 = -s2*temp3 + c2*temp2
						a.Set(k-1, j+i+2-1, c1*temp2+cmplx.Conj(s1)*temp1)
						temp1 = -s1*temp2 + c1*temp1
						a.Set(k-1, j+i+1-1, ctemp*temp1+cmplx.Conj(s)*temp)
						a.Set(k-1, j+i-1, -s*temp1+ctemp*temp)
					}
				}
				//
				if jj > 0 {
					for i = jj; i >= 1; i-- {
						c = a.GetRe(j+1+i-1, j-1)
						Zrot(toPtr((*ihi)-top), a.CVector(top+1-1, j+i+1-1), func() *int { y := 1; return &y }(), a.CVector(top+1-1, j+i-1), func() *int { y := 1; return &y }(), &c, toPtrc128(-b.GetConj(j+1+i-1, j-1)))
					}
				}

				//              Update (J+1)th column of A by transformations from left.
				if j < jcol+nnb-1 {
					len = 1 + j - jcol

					//                 Multiply with the trailing accumulated unitary
					//                 matrix, which takes the form
					//
					//                        [  U11  U12  ]
					//                    U = [            ],
					//                        [  U21  U22  ]
					//
					//                 where U21 is a LEN-by-LEN matrix and U12 is lower
					//                 triangular.
					jrow = (*ihi) - nblst + 1
					err = goblas.Zgemv(ConjTrans, nblst, len, cone, work.CMatrix(nblst, opts), nblst, a.CVector(jrow-1, j+1-1), 1, czero, work.Off(pw-1), 1)
					ppw = pw + len
					for i = jrow; i <= jrow+nblst-len-1; i++ {
						work.Set(ppw-1, a.Get(i-1, j+1-1))
						ppw = ppw + 1
					}
					err = goblas.Ztrmv(Lower, ConjTrans, NonUnit, nblst-len, work.CMatrixOff(len*nblst+1-1, nblst, opts), nblst, work.Off(pw+len-1), 1)
					err = goblas.Zgemv(ConjTrans, len, nblst-len, cone, work.CMatrixOff((len+1)*nblst-len+1-1, nblst, opts), nblst, a.CVector(jrow+nblst-len-1, j+1-1), 1, cone, work.Off(pw+len-1), 1)
					ppw = pw
					for i = jrow; i <= jrow+nblst-1; i++ {
						a.Set(i-1, j+1-1, work.Get(ppw-1))
						ppw = ppw + 1
					}

					//                 Multiply with the other accumulated unitary
					//                 matrices, which take the form
					//
					//                        [  U11  U12   0  ]
					//                        [                ]
					//                    U = [  U21  U22   0  ],
					//                        [                ]
					//                        [   0    0    I  ]
					//
					//                 where I denotes the (NNB-LEN)-by-(NNB-LEN) identity
					//                 matrix, U21 is a LEN-by-LEN upper triangular matrix
					//                 and U12 is an NNB-by-NNB lower triangular matrix.
					ppwo = 1 + nblst*nblst
					j0 = jrow - nnb
					for jrow = j0; jrow >= jcol+1; jrow -= nnb {
						ppw = pw + len
						for i = jrow; i <= jrow+nnb-1; i++ {
							work.Set(ppw-1, a.Get(i-1, j+1-1))
							ppw = ppw + 1
						}
						ppw = pw
						for i = jrow + nnb; i <= jrow+nnb+len-1; i++ {
							work.Set(ppw-1, a.Get(i-1, j+1-1))
							ppw = ppw + 1
						}
						err = goblas.Ztrmv(Upper, ConjTrans, NonUnit, len, work.CMatrixOff(ppwo+nnb-1, 2*nnb, opts), 2*nnb, work.Off(pw-1), 1)
						err = goblas.Ztrmv(Lower, ConjTrans, NonUnit, nnb, work.CMatrixOff(ppwo+2*len*nnb-1, 2*nnb, opts), 2*nnb, work.Off(pw+len-1), 1)
						err = goblas.Zgemv(ConjTrans, nnb, len, cone, work.CMatrixOff(ppwo-1, 2*nnb, opts), 2*nnb, a.CVector(jrow-1, j+1-1), 1, cone, work.Off(pw-1), 1)
						err = goblas.Zgemv(ConjTrans, len, nnb, cone, work.CMatrixOff(ppwo+2*len*nnb+nnb-1, 2*nnb, opts), 2*nnb, a.CVector(jrow+nnb-1, j+1-1), 1, cone, work.Off(pw+len-1), 1)
						ppw = pw
						for i = jrow; i <= jrow+len+nnb-1; i++ {
							a.Set(i-1, j+1-1, work.Get(ppw-1))
							ppw = ppw + 1
						}
						ppwo = ppwo + 4*nnb*nnb
					}
				}
			}

			//           Apply accumulated unitary matrices to A.
			cola = (*n) - jcol - nnb + 1
			j = (*ihi) - nblst + 1
			err = goblas.Zgemm(ConjTrans, NoTrans, nblst, cola, nblst, cone, work.CMatrix(nblst, opts), nblst, a.Off(j-1, jcol+nnb-1), *lda, czero, work.CMatrixOff(pw-1, nblst, opts), nblst)
			Zlacpy('A', &nblst, &cola, work.CMatrixOff(pw-1, nblst, opts), &nblst, a.Off(j-1, jcol+nnb-1), lda)
			ppwo = nblst*nblst + 1
			j0 = j - nnb
			for j = j0; j >= jcol+1; j -= nnb {
				if blk22 {
					//                 Exploit the structure of
					//
					//                        [  U11  U12  ]
					//                    U = [            ]
					//                        [  U21  U22  ],
					//
					//                 where all blocks are NNB-by-NNB, U21 is upper
					//                 triangular and U12 is lower triangular.
					Zunm22('L', 'C', toPtr(2*nnb), &cola, &nnb, &nnb, work.CMatrixOff(ppwo-1, 2*nnb, opts), toPtr(2*nnb), a.Off(j-1, jcol+nnb-1), lda, work.Off(pw-1), toPtr((*lwork)-pw+1), &ierr)
				} else {
					//                 Ignore the structure of U.
					err = goblas.Zgemm(ConjTrans, NoTrans, 2*nnb, cola, 2*nnb, cone, work.CMatrixOff(ppwo-1, 2*nnb, opts), 2*nnb, a.Off(j-1, jcol+nnb-1), *lda, czero, work.CMatrixOff(pw-1, 2*nnb, opts), 2*nnb)
					Zlacpy('A', toPtr(2*nnb), &cola, work.CMatrixOff(pw-1, 2*nnb, opts), toPtr(2*nnb), a.Off(j-1, jcol+nnb-1), lda)
				}
				ppwo = ppwo + 4*nnb*nnb
			}

			//           Apply accumulated unitary matrices to Q.
			if wantq {
				j = (*ihi) - nblst + 1
				if initq {
					topq = maxint(2, j-jcol+1)
					nh = (*ihi) - topq + 1
				} else {
					topq = 1
					nh = (*n)
				}
				err = goblas.Zgemm(NoTrans, NoTrans, nh, nblst, nblst, cone, q.Off(topq-1, j-1), *ldq, work.CMatrix(nblst, opts), nblst, czero, work.CMatrixOff(pw-1, nh, opts), nh)
				Zlacpy('A', &nh, &nblst, work.CMatrixOff(pw-1, nh, opts), &nh, q.Off(topq-1, j-1), ldq)
				ppwo = nblst*nblst + 1
				j0 = j - nnb
				for j = j0; j >= jcol+1; j -= nnb {
					if initq {
						topq = maxint(2, j-jcol+1)
						nh = (*ihi) - topq + 1
					}
					if blk22 {
						//                    Exploit the structure of U.
						Zunm22('R', 'N', &nh, toPtr(2*nnb), &nnb, &nnb, work.CMatrixOff(ppwo-1, 2*nnb, opts), toPtr(2*nnb), q.Off(topq-1, j-1), ldq, work.Off(pw-1), toPtr((*lwork)-pw+1), &ierr)
					} else {
						//                    Ignore the structure of U.
						err = goblas.Zgemm(NoTrans, NoTrans, nh, 2*nnb, 2*nnb, cone, q.Off(topq-1, j-1), *ldq, work.CMatrixOff(ppwo-1, 2*nnb, opts), 2*nnb, czero, work.CMatrixOff(pw-1, nh, opts), nh)
						Zlacpy('A', &nh, toPtr(2*nnb), work.CMatrixOff(pw-1, nh, opts), &nh, q.Off(topq-1, j-1), ldq)
					}
					ppwo = ppwo + 4*nnb*nnb
				}
			}

			//           Accumulate right Givens rotations if required.
			if wantz || top > 0 {
				//              Initialize small unitary factors that will hold the
				//              accumulated Givens rotations in workspace.
				Zlaset('A', &nblst, &nblst, &czero, &cone, work.CMatrix(nblst, opts), &nblst)
				pw = nblst*nblst + 1
				for i = 1; i <= n2nb; i++ {
					Zlaset('A', toPtr(2*nnb), toPtr(2*nnb), &czero, &cone, work.CMatrixOff(pw-1, 2*nnb, opts), toPtr(2*nnb))
					pw = pw + 4*nnb*nnb
				}

				//              Accumulate Givens rotations into workspace array.
				for j = jcol; j <= jcol+nnb-1; j++ {
					ppw = (nblst+1)*(nblst-2) - j + jcol + 1
					len = 2 + j - jcol
					jrow = j + n2nb*nnb + 2
					for i = (*ihi); i >= jrow; i-- {
						ctemp = a.Get(i-1, j-1)
						a.Set(i-1, j-1, czero)
						s = b.Get(i-1, j-1)
						b.Set(i-1, j-1, czero)
						for jj = ppw; jj <= ppw+len-1; jj++ {
							temp = work.Get(jj + nblst - 1)
							work.Set(jj+nblst-1, ctemp*temp-cmplx.Conj(s)*work.Get(jj-1))
							work.Set(jj-1, s*temp+ctemp*work.Get(jj-1))
						}
						len = len + 1
						ppw = ppw - nblst - 1
					}

					ppwo = nblst*nblst + (nnb+j-jcol-1)*2*nnb + nnb
					j0 = jrow - nnb
					for jrow = j0; jrow >= j+2; jrow -= nnb {
						ppw = ppwo
						len = 2 + j - jcol
						for i = jrow + nnb - 1; i >= jrow; i-- {
							ctemp = a.Get(i-1, j-1)
							a.Set(i-1, j-1, czero)
							s = b.Get(i-1, j-1)
							b.Set(i-1, j-1, czero)
							for jj = ppw; jj <= ppw+len-1; jj++ {
								temp = work.Get(jj + 2*nnb - 1)
								work.Set(jj+2*nnb-1, ctemp*temp-cmplx.Conj(s)*work.Get(jj-1))
								work.Set(jj-1, s*temp+ctemp*work.Get(jj-1))
							}
							len = len + 1
							ppw = ppw - 2*nnb - 1
						}
						ppwo = ppwo + 4*nnb*nnb
					}
				}
			} else {

				Zlaset('L', toPtr((*ihi)-jcol-1), &nnb, &czero, &czero, a.Off(jcol+2-1, jcol-1), lda)
				Zlaset('L', toPtr((*ihi)-jcol-1), &nnb, &czero, &czero, b.Off(jcol+2-1, jcol-1), ldb)
			}

			//           Apply accumulated unitary matrices to A and B.
			if top > 0 {
				j = (*ihi) - nblst + 1
				err = goblas.Zgemm(NoTrans, NoTrans, top, nblst, nblst, cone, a.Off(0, j-1), *lda, work.CMatrix(nblst, opts), nblst, czero, work.CMatrixOff(pw-1, top, opts), top)
				Zlacpy('A', &top, &nblst, work.CMatrixOff(pw-1, top, opts), &top, a.Off(0, j-1), lda)
				ppwo = nblst*nblst + 1
				j0 = j - nnb
				for j = j0; j >= jcol+1; j -= nnb {
					if blk22 {
						//                    Exploit the structure of U.
						Zunm22('R', 'N', &top, toPtr(2*nnb), &nnb, &nnb, work.CMatrixOff(ppwo-1, 2*nnb, opts), toPtr(2*nnb), a.Off(0, j-1), lda, work.Off(pw-1), toPtr((*lwork)-pw+1), &ierr)
					} else {
						//                    Ignore the structure of U.
						err = goblas.Zgemm(NoTrans, NoTrans, top, 2*nnb, 2*nnb, cone, a.Off(0, j-1), *lda, work.CMatrixOff(ppwo-1, 2*nnb, opts), 2*nnb, czero, work.CMatrixOff(pw-1, top, opts), top)
						Zlacpy('A', &top, toPtr(2*nnb), work.CMatrixOff(pw-1, top, opts), &top, a.Off(0, j-1), lda)
					}
					ppwo = ppwo + 4*nnb*nnb
				}
				//
				j = (*ihi) - nblst + 1
				err = goblas.Zgemm(NoTrans, NoTrans, top, nblst, nblst, cone, b.Off(0, j-1), *ldb, work.CMatrix(nblst, opts), nblst, czero, work.CMatrixOff(pw-1, top, opts), top)
				Zlacpy('A', &top, &nblst, work.CMatrixOff(pw-1, top, opts), &top, b.Off(0, j-1), ldb)
				ppwo = nblst*nblst + 1
				j0 = j - nnb
				for j = j0; j >= jcol+1; j -= nnb {
					if blk22 {
						//                    Exploit the structure of U.
						Zunm22('R', 'N', &top, toPtr(2*nnb), &nnb, &nnb, work.CMatrixOff(ppwo-1, 2*nnb, opts), toPtr(2*nnb), b.Off(0, j-1), ldb, work.Off(pw-1), toPtr((*lwork)-pw+1), &ierr)
					} else {
						//                    Ignore the structure of U.
						err = goblas.Zgemm(NoTrans, NoTrans, top, 2*nnb, 2*nnb, cone, b.Off(0, j-1), *ldb, work.CMatrixOff(ppwo-1, 2*nnb, opts), 2*nnb, czero, work.CMatrixOff(pw-1, top, opts), top)
						Zlacpy('A', &top, toPtr(2*nnb), work.CMatrixOff(pw-1, top, opts), &top, b.Off(0, j-1), ldb)
					}
					ppwo = ppwo + 4*nnb*nnb
				}
			}

			//           Apply accumulated unitary matrices to Z.
			if wantz {
				j = (*ihi) - nblst + 1
				if initq {
					topq = maxint(2, j-jcol+1)
					nh = (*ihi) - topq + 1
				} else {
					topq = 1
					nh = (*n)
				}
				err = goblas.Zgemm(NoTrans, NoTrans, nh, nblst, nblst, cone, z.Off(topq-1, j-1), *ldz, work.CMatrix(nblst, opts), nblst, czero, work.CMatrixOff(pw-1, nh, opts), nh)
				Zlacpy('A', &nh, &nblst, work.CMatrixOff(pw-1, nh, opts), &nh, z.Off(topq-1, j-1), ldz)
				ppwo = nblst*nblst + 1
				j0 = j - nnb
				for j = j0; j >= jcol+1; j -= nnb {
					if initq {
						topq = maxint(2, j-jcol+1)
						nh = (*ihi) - topq + 1
					}
					if blk22 {
						//                    Exploit the structure of U.
						Zunm22('R', 'N', &nh, toPtr(2*nnb), &nnb, &nnb, work.CMatrixOff(ppwo-1, 2*nnb, opts), toPtr(2*nnb), z.Off(topq-1, j-1), ldz, work.Off(pw-1), toPtr((*lwork)-pw+1), &ierr)
					} else {
						//                    Ignore the structure of U.
						err = goblas.Zgemm(NoTrans, NoTrans, nh, 2*nnb, 2*nnb, cone, z.Off(topq-1, j-1), *ldz, work.CMatrixOff(ppwo-1, 2*nnb, opts), 2*nnb, czero, work.CMatrixOff(pw-1, nh, opts), nh)
						Zlacpy('A', &nh, toPtr(2*nnb), work.CMatrixOff(pw-1, nh, opts), &nh, z.Off(topq-1, j-1), ldz)
					}
					ppwo = ppwo + 4*nnb*nnb
				}
			}
		}
	}

	//     Use unblocked code to reduce the rest of the matrix
	//     Avoid re-initialization of modified Q and Z.
	compq2 = compq
	compz2 = compz
	if jcol != (*ilo) {
		if wantq {
			compq2 = 'V'
		}
		if wantz {
			compz2 = 'V'
		}
	}

	if jcol < (*ihi) {
		Zgghrd(compq2, compz2, n, &jcol, ihi, a, lda, b, ldb, q, ldq, z, ldz, &ierr)
	}
	work.SetRe(0, float64(lwkopt))
}
