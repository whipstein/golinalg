package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dgghd3 reduces a pair of real matrices (A,B) to generalized upper
// Hessenberg form using orthogonal transformations, where A is a
// general matrix and B is upper triangular.  The form of the
// generalized eigenvalue problem is
//    A*x = lambda*B*x,
// and B is typically made upper triangular by computing its QR
// factorization and moving the orthogonal matrix Q to the left side
// of the equation.
//
// This subroutine simultaneously reduces A to a Hessenberg matrix H:
//    Q**T*A*Z = H
// and transforms B to another upper triangular matrix T:
//    Q**T*B*Z = T
// in order to reduce the problem to its standard form
//    H*y = lambda*T*y
// where y = Z**T*x.
//
// The orthogonal matrices Q and Z are determined as products of Givens
// rotations.  They may either be formed explicitly, or they may be
// postmultiplied into input matrices Q1 and Z1, so that
//
//      Q1 * A * Z1**T = (Q1*Q) * H * (Z1*Z)**T
//
//      Q1 * B * Z1**T = (Q1*Q) * T * (Z1*Z)**T
//
// If Q1 is the orthogonal matrix from the QR factorization of B in the
// original equation A*x = lambda*B*x, then Dgghd3 reduces the original
// problem to generalized Hessenberg form.
//
// This is a blocked variant of DGGHRD, using matrix-matrix
// multiplications for parts of the computation to enhance performance.
func Dgghd3(compq, compz byte, n, ilo, ihi int, a, b, q, z *mat.Matrix, work *mat.Vector, lwork int) (err error) {
	var blk22, initq, initz, lquery, wantq, wantz bool
	var compq2, compz2 byte
	var c, c1, c2, one, s, s1, s2, temp, temp1, temp2, temp3, zero float64
	var cola, i, j, j0, jcol, jj, jrow, k, kacc22, len, lwkopt, n2nb, nb, nblst, nbmin, nh, nnb, nx, ppw, ppwo, pw, top, topq int

	zero = 0.0
	one = 1.0

	//     Decode and test the input parameters.
	nb = Ilaenv(1, "Dgghd3", []byte{' '}, n, ilo, ihi, -1)
	lwkopt = max(6*n*nb, 1)
	work.Set(0, float64(lwkopt))
	initq = compq == 'I'
	wantq = initq || compq == 'V'
	initz = compz == 'I'
	wantz = initz || compz == 'V'
	lquery = (lwork == -1)

	if compq != 'N' && !wantq {
		err = fmt.Errorf("compq != 'N' && !wantq: compq='%c'", compq)
	} else if compz != 'N' && !wantz {
		err = fmt.Errorf("compz != 'N' && !wantz: compz='%c'", compz)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if ilo < 1 {
		err = fmt.Errorf("ilo < 1: ilo=%v", ilo)
	} else if ihi > n || ihi < ilo-1 {
		err = fmt.Errorf("ihi > n || ihi < ilo-1: ilo=%v, ihi=%v", ilo, ihi)
	} else if a.Rows < max(1, n) {
		err = fmt.Errorf("a.Rows < max(1, n): a.Rows=%v, n=%v", a.Rows, n)
	} else if b.Rows < max(1, n) {
		err = fmt.Errorf("b.Rows < max(1, n): b.Rows=%v, n=%v", b.Rows, n)
	} else if (wantq && q.Rows < n) || q.Rows < 1 {
		err = fmt.Errorf("(wantq && q.Rows < n) || q.Rows < 1: compq='%c', q.Rows=%v, n=%v", compq, q.Rows, n)
	} else if (wantz && z.Rows < n) || z.Rows < 1 {
		err = fmt.Errorf("(wantz && z.Rows < n) || z.Rows < 1: compz='%c', z.Rows=%v, n=%v", compz, z.Rows, n)
	} else if lwork < 1 && !lquery {
		err = fmt.Errorf("lwork < 1 && !lquery: lwork=%v, lquery=%v", lwork, lquery)
	}
	if err != nil {
		gltest.Xerbla2("Dgghd3", err)
		return
	} else if lquery {
		return
	}

	//     Initialize Q and Z if desired.
	if initq {
		Dlaset(Full, n, n, zero, one, q)
	}
	if initz {
		Dlaset(Full, n, n, zero, one, z)
	}

	//     Zero out lower triangle of B.
	if n > 1 {
		Dlaset(Lower, n-1, n-1, zero, zero, b.Off(1, 0))
	}

	//     Quick return if possible
	nh = ihi - ilo + 1
	if nh <= 1 {
		work.Set(0, one)
		return
	}

	//     Determine the blocksize.
	nbmin = Ilaenv(2, "Dgghd3", []byte{' '}, n, ilo, ihi, -1)
	if nb > 1 && nb < nh {
		//        Determine when to use unblocked instead of blocked code.
		nx = max(nb, Ilaenv(3, "Dgghd3", []byte{' '}, n, ilo, ihi, -1))
		if nx < nh {
			//           Determine if workspace is large enough for blocked code.
			if lwork < lwkopt {
				//              Not enough workspace to use optimal NB:  determine the
				//              minimum value of NB, and reduce NB or force use of
				//              unblocked code.
				nbmin = max(2, Ilaenv(2, "Dgghd3", []byte{' '}, n, ilo, ihi, -1))
				if lwork >= 6*n*nbmin {
					nb = lwork / (6 * n)
				} else {
					nb = 1
				}
			}
		}
	}

	if nb < nbmin || nb >= nh {
		//        Use unblocked code below
		jcol = ilo

	} else {
		//        Use blocked code
		kacc22 = Ilaenv(16, "Dgghd3", []byte{' '}, n, ilo, ihi, -1)
		blk22 = kacc22 == 2
		for _, jcol = range genIter(ilo, ihi-2, nb) {
			nnb = min(nb, ihi-jcol-1)

			//           Initialize small orthogonal factors that will hold the
			//           accumulated Givens rotations in workspace.
			//           N2NB   denotes the number of 2*NNB-by-2*NNB factors
			//           NBLST  denotes the (possibly smaller) order of the last
			//                  factor.
			n2nb = (ihi-jcol-1)/nnb - 1
			nblst = ihi - jcol - n2nb*nnb
			Dlaset(Full, nblst, nblst, zero, one, work.Matrix(nblst, opts))
			pw = nblst*nblst + 1
			for i = 1; i <= n2nb; i++ {
				Dlaset(Full, 2*nnb, 2*nnb, zero, one, work.Off(pw-1).Matrix(2*nnb, opts))
				pw = pw + 4*nnb*nnb
			}

			//           Reduce columns JCOL:JCOL+NNB-1 of A to Hessenberg form.
			for j = jcol; j <= jcol+nnb-1; j++ {
				//              Reduce Jth column of A. Store cosines and sines in Jth
				//              column of A and B, respectively.
				for i = ihi; i >= j+2; i-- {
					temp = a.Get(i-1-1, j-1)
					c, s, *a.GetPtr(i-1-1, j-1) = Dlartg(temp, a.Get(i-1, j-1))
					a.Set(i-1, j-1, c)
					b.Set(i-1, j-1, s)
				}

				//              Accumulate Givens rotations into workspace array.
				ppw = (nblst+1)*(nblst-2) - j + jcol + 1
				len = 2 + j - jcol
				jrow = j + n2nb*nnb + 2
				for i = ihi; i >= jrow; i-- {
					c = a.Get(i-1, j-1)
					s = b.Get(i-1, j-1)
					for jj = ppw; jj <= ppw+len-1; jj++ {
						temp = work.Get(jj + nblst - 1)
						work.Set(jj+nblst-1, c*temp-s*work.Get(jj-1))
						work.Set(jj-1, s*temp+c*work.Get(jj-1))
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
						c = a.Get(i-1, j-1)
						s = b.Get(i-1, j-1)
						for jj = ppw; jj <= ppw+len-1; jj++ {
							temp = work.Get(jj + 2*nnb - 1)
							work.Set(jj+2*nnb-1, c*temp-s*work.Get(jj-1))
							work.Set(jj-1, s*temp+c*work.Get(jj-1))
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
				for jj = n; jj >= j+1; jj-- {
					//                 Update JJth column of B.
					for i = min(jj+1, ihi); i >= j+2; i-- {
						c = a.Get(i-1, j-1)
						s = b.Get(i-1, j-1)
						temp = b.Get(i-1, jj-1)
						b.Set(i-1, jj-1, c*temp-s*b.Get(i-1-1, jj-1))
						b.Set(i-1-1, jj-1, s*temp+c*b.Get(i-1-1, jj-1))
					}

					//                 Annihilate B( JJ+1, JJ ).
					if jj < ihi {
						temp = b.Get(jj, jj)
						c, s, *b.GetPtr(jj, jj) = Dlartg(temp, b.Get(jj, jj-1))
						b.Set(jj, jj-1, zero)
						b.Off(top, jj-1).Vector().Rot(jj-top, b.Off(top, jj).Vector(), 1, 1, c, s)
						a.Set(jj, j-1, c)
						b.Set(jj, j-1, -s)
					}
				}

				//              Update A by transformations from right.
				//              Explicit loop unrolling provides better performance
				//              compared to DLASR.
				//               CALL DLASR( 'Right', 'Variable', 'Backward', IHI-TOP,
				//     $                     IHI-J, A( J+2, J ), B( J+2, J ),
				//     $                     A( TOP+1, J+1 ), LDA )
				jj = (ihi - j - 1) % 3
				for i = ihi - j - 3; i >= jj+1; i -= 3 {
					c = a.Get(j+1+i-1, j-1)
					s = -b.Get(j+1+i-1, j-1)
					c1 = a.Get(j+2+i-1, j-1)
					s1 = -b.Get(j+2+i-1, j-1)
					c2 = a.Get(j+3+i-1, j-1)
					s2 = -b.Get(j+3+i-1, j-1)

					for k = top + 1; k <= ihi; k++ {
						temp = a.Get(k-1, j+i-1)
						temp1 = a.Get(k-1, j+i)
						temp2 = a.Get(k-1, j+i+2-1)
						temp3 = a.Get(k-1, j+i+3-1)
						a.Set(k-1, j+i+3-1, c2*temp3+s2*temp2)
						temp2 = -s2*temp3 + c2*temp2
						a.Set(k-1, j+i+2-1, c1*temp2+s1*temp1)
						temp1 = -s1*temp2 + c1*temp1
						a.Set(k-1, j+i, c*temp1+s*temp)
						a.Set(k-1, j+i-1, -s*temp1+c*temp)
					}
				}

				if jj > 0 {
					for i = jj; i >= 1; i-- {
						a.Off(top, j+i-1).Vector().Rot(ihi-top, a.Off(top, j+i).Vector(), 1, 1, a.Get(j+1+i-1, j-1), -b.Get(j+1+i-1, j-1))
					}
				}

				//              Update (J+1)th column of A by transformations from left.
				if j < jcol+nnb-1 {
					len = 1 + j - jcol

					//                 Multiply with the trailing accumulated orthogonal
					//                 matrix, which takes the form
					//
					//                        [  U11  U12  ]
					//                    U = [            ],
					//                        [  U21  U22  ]
					//
					//                 where U21 is a LEN-by-LEN matrix and U12 is lower
					//                 triangular.
					jrow = ihi - nblst + 1
					err = work.Off(pw-1).Gemv(Trans, nblst, len, one, work.Matrix(nblst, opts), a.Off(jrow-1, j).Vector(), 1, zero, 1)
					ppw = pw + len
					for i = jrow; i <= jrow+nblst-len-1; i++ {
						work.Set(ppw-1, a.Get(i-1, j))
						ppw = ppw + 1
					}
					err = work.Off(pw+len-1).Trmv(Lower, NoTrans, NonUnit, nblst-len, work.Off(len*nblst).Matrix(nblst, opts), 1)
					err = work.Off(pw+len-1).Gemv(Trans, len, nblst-len, one, work.Off((len+1)*nblst-len).Matrix(nblst, opts), a.Off(jrow+nblst-len-1, j).Vector(), 1, one, 1)
					ppw = pw
					for i = jrow; i <= jrow+nblst-1; i++ {
						a.Set(i-1, j, work.Get(ppw-1))
						ppw = ppw + 1
					}

					//                 Multiply with the other accumulated orthogonal
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
							work.Set(ppw-1, a.Get(i-1, j))
							ppw = ppw + 1
						}
						ppw = pw
						for i = jrow + nnb; i <= jrow+nnb+len-1; i++ {
							work.Set(ppw-1, a.Get(i-1, j))
							ppw = ppw + 1
						}
						err = work.Off(pw-1).Trmv(Upper, NoTrans, NonUnit, len, work.Off(ppwo+nnb-1).Matrix(2*nnb, opts), 1)
						err = work.Off(pw+len-1).Trmv(Lower, NoTrans, NonUnit, nnb, work.Off(ppwo+2*len*nnb-1).Matrix(2*nnb, opts), 1)
						err = work.Off(pw-1).Gemv(Trans, nnb, len, one, work.Off(ppwo-1).Matrix(2*nnb, opts), a.Off(jrow-1, j).Vector(), 1, one, 1)
						err = work.Off(pw+len-1).Gemv(Trans, len, nnb, one, work.Off(ppwo+2*len*nnb+nnb-1).Matrix(2*nnb, opts), a.Off(jrow+nnb-1, j).Vector(), 1, one, 1)
						ppw = pw
						for i = jrow; i <= jrow+len+nnb-1; i++ {
							a.Set(i-1, j, work.Get(ppw-1))
							ppw = ppw + 1
						}
						ppwo = ppwo + 4*nnb*nnb
					}
				}
			}

			//           Apply accumulated orthogonal matrices to A.
			cola = n - jcol - nnb + 1
			j = ihi - nblst + 1
			err = work.Off(pw-1).Matrix(nblst, opts).Gemm(Trans, NoTrans, nblst, cola, nblst, one, work.Matrix(nblst, opts), a.Off(j-1, jcol+nnb-1), zero)
			Dlacpy(Full, nblst, cola, work.Off(pw-1).Matrix(nblst, opts), a.Off(j-1, jcol+nnb-1))
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
					if err = Dorm22(Left, Trans, 2*nnb, cola, nnb, nnb, work.Off(ppwo-1).Matrix(2*nnb, opts), a.Off(j-1, jcol+nnb-1), work.Off(pw-1), lwork-pw+1); err != nil {
						panic(err)
					}
				} else {
					//                 Ignore the structure of U.
					err = work.Off(pw-1).Matrix(2*nnb, opts).Gemm(Trans, NoTrans, 2*nnb, cola, 2*nnb, one, work.Off(ppwo-1).Matrix(2*nnb, opts), a.Off(j-1, jcol+nnb-1), zero)
					Dlacpy(Full, 2*nnb, cola, work.Off(pw-1).Matrix(2*nnb, opts), a.Off(j-1, jcol+nnb-1))
				}
				ppwo = ppwo + 4*nnb*nnb
			}

			//           Apply accumulated orthogonal matrices to Q.
			if wantq {
				j = ihi - nblst + 1
				if initq {
					topq = max(2, j-jcol+1)
					nh = ihi - topq + 1
				} else {
					topq = 1
					nh = n
				}
				err = work.Off(pw-1).Matrix(nh, opts).Gemm(NoTrans, NoTrans, nh, nblst, nblst, one, q.Off(topq-1, j-1), work.Matrix(nblst, opts), zero)
				Dlacpy(Full, nh, nblst, work.Off(pw-1).Matrix(nh, opts), q.Off(topq-1, j-1))
				ppwo = nblst*nblst + 1
				j0 = j - nnb
				for j = j0; j >= jcol+1; j -= nnb {
					if initq {
						topq = max(2, j-jcol+1)
						nh = ihi - topq + 1
					}
					if blk22 {
						//                    Exploit the structure of U.
						if err = Dorm22(Right, NoTrans, nh, 2*nnb, nnb, nnb, work.Off(ppwo-1).Matrix(2*nnb, opts), q.Off(topq-1, j-1), work.Off(pw-1), lwork-pw+1); err != nil {
							panic(err)
						}
					} else {
						//                    Ignore the structure of U.
						err = work.Off(pw-1).Matrix(nh, opts).Gemm(NoTrans, NoTrans, nh, 2*nnb, 2*nnb, one, q.Off(topq-1, j-1), work.Off(ppwo-1).Matrix(2*nnb, opts), zero)
						Dlacpy(Full, nh, 2*nnb, work.Off(pw-1).Matrix(nh, opts), q.Off(topq-1, j-1))
					}
					ppwo = ppwo + 4*nnb*nnb
				}
			}

			//           Accumulate right Givens rotations if required.
			if wantz || top > 0 {
				//              Initialize small orthogonal factors that will hold the
				//              accumulated Givens rotations in workspace.
				Dlaset(Full, nblst, nblst, zero, one, work.Matrix(nblst, opts))
				pw = nblst*nblst + 1
				for i = 1; i <= n2nb; i++ {
					Dlaset(Full, 2*nnb, 2*nnb, zero, one, work.Off(pw-1).Matrix(2*nnb, opts))
					pw = pw + 4*nnb*nnb
				}

				//              Accumulate Givens rotations into workspace array.
				for j = jcol; j <= jcol+nnb-1; j++ {
					ppw = (nblst+1)*(nblst-2) - j + jcol + 1
					len = 2 + j - jcol
					jrow = j + n2nb*nnb + 2
					for i = ihi; i >= jrow; i-- {
						c = a.Get(i-1, j-1)
						a.Set(i-1, j-1, zero)
						s = b.Get(i-1, j-1)
						b.Set(i-1, j-1, zero)
						for jj = ppw; jj <= ppw+len-1; jj++ {
							temp = work.Get(jj + nblst - 1)
							work.Set(jj+nblst-1, c*temp-s*work.Get(jj-1))
							work.Set(jj-1, s*temp+c*work.Get(jj-1))
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
							c = a.Get(i-1, j-1)
							a.Set(i-1, j-1, zero)
							s = b.Get(i-1, j-1)
							b.Set(i-1, j-1, zero)
							for jj = ppw; jj <= ppw+len-1; jj++ {
								temp = work.Get(jj + 2*nnb - 1)
								work.Set(jj+2*nnb-1, c*temp-s*work.Get(jj-1))
								work.Set(jj-1, s*temp+c*work.Get(jj-1))
							}
							len = len + 1
							ppw = ppw - 2*nnb - 1
						}
						ppwo = ppwo + 4*nnb*nnb
					}
				}
			} else {

				Dlaset(Lower, ihi-jcol-1, nnb, zero, zero, a.Off(jcol+2-1, jcol-1))
				Dlaset(Lower, ihi-jcol-1, nnb, zero, zero, b.Off(jcol+2-1, jcol-1))
			}

			//           Apply accumulated orthogonal matrices to A and B.
			if top > 0 {
				j = ihi - nblst + 1
				err = work.Off(pw-1).Matrix(top, opts).Gemm(NoTrans, NoTrans, top, nblst, nblst, one, a.Off(0, j-1), work.Matrix(nblst, opts), zero)
				Dlacpy(Full, top, nblst, work.Off(pw-1).Matrix(top, opts), a.Off(0, j-1))
				ppwo = nblst*nblst + 1
				j0 = j - nnb
				for j = j0; j >= jcol+1; j -= nnb {
					if blk22 {
						//                    Exploit the structure of U.
						if err = Dorm22(Right, NoTrans, top, 2*nnb, nnb, nnb, work.Off(ppwo-1).Matrix(2*nnb, opts), a.Off(0, j-1), work.Off(pw-1), lwork-pw+1); err != nil {
							panic(err)
						}
					} else {
						//                    Ignore the structure of U.
						work.Off(pw-1).Matrix(top, opts).Gemm(NoTrans, NoTrans, top, 2*nnb, 2*nnb, one, a.Off(0, j-1), work.Off(ppwo-1).Matrix(2*nnb, opts), zero)
						Dlacpy(Full, top, 2*nnb, work.Off(pw-1).Matrix(top, opts), a.Off(0, j-1))
					}
					ppwo = ppwo + 4*nnb*nnb
				}

				j = ihi - nblst + 1
				err = work.Off(pw-1).Matrix(top, opts).Gemm(NoTrans, NoTrans, top, nblst, nblst, one, b.Off(0, j-1), work.Matrix(nblst, opts), zero)
				Dlacpy(Full, top, nblst, work.Off(pw-1).Matrix(top, opts), b.Off(0, j-1))
				ppwo = nblst*nblst + 1
				j0 = j - nnb
				for j = j0; j >= jcol+1; j -= nnb {
					if blk22 {
						//                    Exploit the structure of U.
						if err = Dorm22(Right, NoTrans, top, 2*nnb, nnb, nnb, work.Off(ppwo-1).Matrix(2*nnb, opts), b.Off(0, j-1), work.Off(pw-1), lwork-pw+1); err != nil {
							panic(err)
						}
					} else {
						//                    Ignore the structure of U.
						err = work.Off(pw-1).Matrix(top, opts).Gemm(NoTrans, NoTrans, top, 2*nnb, 2*nnb, one, b.Off(0, j-1), work.Off(ppwo-1).Matrix(2*nnb, opts), zero)
						Dlacpy(Full, top, 2*nnb, work.Off(pw-1).Matrix(top, opts), b.Off(0, j-1))
					}
					ppwo = ppwo + 4*nnb*nnb
				}
			}

			//           Apply accumulated orthogonal matrices to Z.
			if wantz {
				j = ihi - nblst + 1
				if initq {
					topq = max(2, j-jcol+1)
					nh = ihi - topq + 1
				} else {
					topq = 1
					nh = n
				}
				err = work.Off(pw-1).Matrix(nh, opts).Gemm(NoTrans, NoTrans, nh, nblst, nblst, one, z.Off(topq-1, j-1), work.Matrix(nblst, opts), zero)
				Dlacpy(Full, nh, nblst, work.Off(pw-1).Matrix(nh, opts), z.Off(topq-1, j-1))
				ppwo = nblst*nblst + 1
				j0 = j - nnb
				for j = j0; j >= jcol+1; j -= nnb {
					if initq {
						topq = max(2, j-jcol+1)
						nh = ihi - topq + 1
					}
					if blk22 {
						//                    Exploit the structure of U.
						if err = Dorm22(Right, NoTrans, nh, 2*nnb, nnb, nnb, work.Off(ppwo-1).Matrix(2*nnb, opts), z.Off(topq-1, j-1), work.Off(pw-1), lwork-pw+1); err != nil {
							panic(err)
						}
					} else {
						//                    Ignore the structure of U.
						err = work.Off(pw-1).Matrix(nh, opts).Gemm(NoTrans, NoTrans, nh, 2*nnb, 2*nnb, one, z.Off(topq-1, j-1), work.Off(ppwo-1).Matrix(2*nnb, opts), zero)
						Dlacpy(Full, nh, 2*nnb, work.Off(pw-1).Matrix(nh, opts), z.Off(topq-1, j-1))
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
	if jcol != ilo {
		if wantq {
			compq2 = 'V'
		}
		if wantz {
			compz2 = 'V'
		}
	}

	if jcol < ihi {
		if err = Dgghrd(compq2, compz2, n, jcol, ihi, a, b, q, z); err != nil {
			panic(err)
		}
	}
	work.Set(0, float64(lwkopt))

	return
}
