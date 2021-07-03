package goblas

import (
	"fmt"
	"math"
	"math/cmplx"
	"testing"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

func TestZblasLevel3(t *testing.T) {
	var diag, diags mat.MatDiag
	var side, sides mat.MatSide
	var tranas, tranbs, transa, transb, trans, transs, transt mat.MatTrans
	var uplo, uplos mat.MatUplo
	var fatal, left, null, reset, trana, tranb, same, upper bool
	var alpha, als, beta, bets, bls, one, zero complex128
	var eps, err, errmax, ralpha, rals, rbeta, rbets, rone, rzero, thresh float64
	var i, ia, ib, ica, icb, icd, ics, ict, icu, ik, im, in, j, jc, jj, jjab, k, ks, laa, lbb, lcc, lda, ldas, ldb, ldbs, ldc, ldcs, lj, m, ma, mb, ms, n, na, nalf, nargs, nb, nbet, nc, nidim, nmax, ns int
	var err2 error
	_ = err2
	ok := &gltest.Common.Infoc.Ok
	*ok = true
	reset = true
	eps = epsilonf64()
	thresh = 16.0
	errmax = 0.0
	nmax = 65
	isame := make([]bool, 13)
	ichd := []mat.MatDiag{Unit, NonUnit}
	ichs := []mat.MatSide{Left, Right}
	icht := []mat.MatTrans{NoTrans, Trans, ConjTrans}
	ichu := []mat.MatUplo{Upper, Lower}
	idim := []int{0, 1, 2, 3, 5, 9}
	alf := cvdf([]complex128{0.0 + 0.0i, 1.0 + 0.0i, 0.7 - 0.9i})
	bet := cvdf([]complex128{0.0 + 0.0i, 1.0 + 0.0i, 1.3 - 1.1i})
	nidim = len(idim)
	nalf = len(alf.Data)
	nbet = len(bet.Data)
	aa := cvf(nmax * nmax)
	ab := cmf(nmax, 2*nmax, opts)
	as := cvf(nmax * nmax)
	bb := cvf(nmax * nmax)
	bs := cvf(nmax * nmax)
	c := cmf(nmax, nmax, opts)
	cc := cvf(nmax * nmax)
	cs := cvf(nmax * nmax)
	ct := cvf(nmax)
	w := cvf(2 * nmax)
	g := vf(nmax)
	fmt.Printf("\n***** ZBLAS Level 3 Tests *****\n")

	zero = (0.0 + 0.0*1i)
	one = (1.0 + 0.0*1i)
	rzero = 0.0
	rone = 1.0
	nmax = 65
	snames := []string{"Zgemm", "Zhemm", "Zsymm", "Ztrmm", "Ztrsm", "Zherk", "Zsyrk", "Zher2k", "Zsyr2k"}

	//     Check the reliability of ZMMCH using exact data.
	n = min(int(32), nmax)
	for j = 1; j <= n; j++ {
		for i = 1; i <= n; i++ {
			ab.SetRe(i-1, j-1, float64(max(i-j+1, 0)))
		}
		ab.SetRe(j-1, nmax+1-1, float64(j))
		ab.SetRe(0, nmax+j-1, float64(j))
		c.Set(j-1, 0, zero)
	}
	for j = 1; j <= n; j++ {
		cc.SetRe(j-1, float64(j*((j+1)*j))/2-float64((j+1)*j*(j-1))/3)
	}
	//     CC holds the exact result. On exit from ZMMCH CT holds
	//     the result computed by ZMMCH.
	transa = NoTrans
	transb = NoTrans
	zmmch(transa, transb, n, 1, n, one, ab, nmax, ab.Off(0, nmax+1-1), nmax, zero, c, nmax, ct, g, cc.CMatrix(nmax, opts), nmax, eps, &err, &fatal, true, t)
	same = lze(cc, ct, n)
	if !same || err != rzero {
		t.Fail()
		fmt.Printf(" ERROR IN ZMMCH -  IN-LINE DOT PRODUCTS ARE BEING EVALUATED WRONGLY.\n ZMMCH WAS CALLED WITH TRANSA = %c AND TRANSB = %c\n AND RETURNED SAME =  %t AND ERR = %12.3f.\n THIS MAY BE DUE TO FAULTS IN THE ARITHMETIC OR THE COMPILER.\n ******* TESTS ABANDONED *******\n", transa, transb, same, err)
	}
	transb = ConjTrans
	zmmch(transa, transb, n, 1, n, one, ab, nmax, ab.Off(0, nmax+1-1), nmax, zero, c, nmax, ct, g, cc.CMatrix(nmax, opts), nmax, eps, &err, &fatal, true, t)
	same = lze(cc, ct, n)
	if !same || err != rzero {
		t.Fail()
		fmt.Printf(" ERROR IN ZMMCH -  IN-LINE DOT PRODUCTS ARE BEING EVALUATED WRONGLY.\n ZMMCH WAS CALLED WITH TRANSA = %c AND TRANSB = %c\n AND RETURNED SAME =  %t AND ERR = %12.3f.\n THIS MAY BE DUE TO FAULTS IN THE ARITHMETIC OR THE COMPILER.\n ******* TESTS ABANDONED *******\n", transa, transb, same, err)
	}
	for j = 1; j <= n; j++ {
		ab.SetRe(j-1, nmax+1-1, float64(n-j+1))
		ab.SetRe(0, nmax+j-1, float64(n-j+1))
	}
	for j = 1; j <= n; j++ {
		cc.SetRe(n-j+1-1, float64(j*((j+1)*j))/2-float64((j+1)*j*(j-1))/3)
	}
	transa = ConjTrans
	transb = NoTrans
	zmmch(transa, transb, n, 1, n, one, ab, nmax, ab.Off(0, nmax+1-1), nmax, zero, c, nmax, ct, g, cc.CMatrix(nmax, opts), nmax, eps, &err, &fatal, true, t)
	same = lze(cc, ct, n)
	if !same || err != rzero {
		t.Fail()
		fmt.Printf(" ERROR IN ZMMCH -  IN-LINE DOT PRODUCTS ARE BEING EVALUATED WRONGLY.\n ZMMCH WAS CALLED WITH TRANSA = %c AND TRANSB = %c\n AND RETURNED SAME =  %t AND ERR = %12.3f.\n THIS MAY BE DUE TO FAULTS IN THE ARITHMETIC OR THE COMPILER.\n ******* TESTS ABANDONED *******\n", transa, transb, same, err)
	}
	transb = ConjTrans
	zmmch(transa, transb, n, 1, n, one, ab, nmax, ab.Off(0, nmax+1-1), nmax, zero, c, nmax, ct, g, cc.CMatrix(nmax, opts), nmax, eps, &err, &fatal, true, t)
	same = lze(cc, ct, n)
	if !same || err != rzero {
		t.Fail()
		fmt.Printf(" ERROR IN ZMMCH -  IN-LINE DOT PRODUCTS ARE BEING EVALUATED WRONGLY.\n ZMMCH WAS CALLED WITH TRANSA = %c AND TRANSB = %c\n AND RETURNED SAME =  %t AND ERR = %12.3f.\n THIS MAY BE DUE TO FAULTS IN THE ARITHMETIC OR THE COMPILER.\n ******* TESTS ABANDONED *******\n", transa, transb, same, err)
	}

	for _, sname := range snames {
		fatal = false
		reset = true
		*ok = true
		errmax = 0.0
		nc = 0
		if sname == "Zgemm" {
			nargs = 13
			zchkeLevel3("Zgemm")

			for im = 1; im <= nidim; im++ {
				m = idim[im-1]

				for in = 1; in <= nidim; in++ {
					n = idim[in-1]
					//           Set LDC to 1 more than minimum value if room.
					ldc = m
					if ldc < nmax {
						ldc = ldc + 1
					}
					//           Skip tests if not enough room.
					if ldc > nmax {
						goto label1100
					}
					lcc = ldc * n
					null = n <= 0 || m <= 0

					for ik = 1; ik <= nidim; ik++ {
						k = idim[ik-1]

						for ica = 1; ica <= 3; ica++ {
							transa = icht[ica-1]
							trana = transa == Trans || transa == ConjTrans

							if trana {
								ma = k
								na = m
							} else {
								ma = m
								na = k
							}
							//                 Set LDA to 1 more than minimum value if room.
							lda = ma
							if lda < nmax {
								lda = lda + 1
							}
							//                 Skip tests if not enough room.
							if lda > nmax {
								goto label180
							}
							laa = lda * na

							//                 Generate the matrix A.
							zmakeL3([]byte("ge"), Full, NonUnit, ma, na, ab, nmax, aa, lda, &reset, zero)

							for icb = 1; icb <= 3; icb++ {
								transb = icht[icb-1]
								tranb = transb == Trans || transb == ConjTrans

								if tranb {
									mb = n
									nb = k
								} else {
									mb = k
									nb = n
								}
								//                    Set LDB to 1 more than minimum value if room.
								ldb = mb
								if ldb < nmax {
									ldb = ldb + 1
								}
								//                    Skip tests if not enough room.
								if ldb > nmax {
									goto label170
								}
								lbb = ldb * nb

								//                    Generate the matrix B.
								zmakeL3([]byte("ge"), Full, NonUnit, mb, nb, ab.Off(0, nmax+1-1), nmax, bb, ldb, &reset, zero)

								for ia = 1; ia <= nalf; ia++ {
									alpha = alf.Get(ia - 1)

									for ib = 1; ib <= nbet; ib++ {
										beta = bet.Get(ib - 1)

										//                          Generate the matrix C.
										zmakeL3([]byte("ge"), Full, NonUnit, m, n, c, nmax, cc, ldc, &reset, zero)

										nc = nc + 1

										//                          Save every datum before calling the
										//                          subroutine.
										tranas = transa
										tranbs = transb
										ms = m
										ns = n
										ks = k
										als = alpha
										as = aa.DeepCopy()
										ldas = lda
										bs = bb.DeepCopy()
										ldbs = ldb
										bls = beta
										cs = cc.DeepCopy()
										ldcs = ldc

										//                          Call the subroutine.
										err2 = Zgemm(transa, transb, m, n, k, alpha, aa.CMatrix(lda, opts), lda, bb.CMatrix(ldb, opts), ldb, beta, cc.CMatrix(ldc, opts), ldc)

										//                          Check if error-exit was taken incorrectly.
										if !(*ok) {
											fmt.Printf(" ******* FATAL ERROR - ERROR-EXIT TAKEN ON VALID CALL *******\n")
											fatal = true
											goto label1120
										}

										//                          See what data changed inside subroutines.
										isame[0] = transa == tranas
										isame[1] = transb == tranbs
										isame[2] = ms == m
										isame[3] = ns == n
										isame[4] = ks == k
										isame[5] = als == alpha
										isame[6] = lze(as, aa, laa)
										isame[7] = ldas == lda
										isame[8] = lze(bs, bb, lbb)
										isame[9] = ldbs == ldb
										isame[10] = bls == beta
										if null {
											isame[11] = lze(cs, cc, lcc)
										} else {
											isame[11] = lzeres([]byte("ge"), Full, m, n, cs.CMatrix(ldc, opts), cc.CMatrix(ldc, opts), ldc)
										}
										isame[12] = ldcs == ldc

										//                          If data was incorrectly changed, report
										//                          and return.
										same = true
										for i = 1; i <= nargs; i++ {
											same = same && isame[i-1]
											if !isame[i-1] {
												t.Fail()
												fmt.Printf(" ******* FATAL ERROR - PARAMETER NUMBER %2d WAS CHANGED INCORRECTLY *******\n", i)
											}
										}
										if !same {
											fatal = true
											goto label1120
										}

										if !null {
											//                             Check the result.
											zmmch(transa, transb, m, n, k, alpha, ab, nmax, ab.Off(0, nmax+1-1), nmax, beta, c, nmax, ct, g, cc.CMatrix(ldc, opts), ldc, eps, &err, &fatal, true, t)
											errmax = math.Max(errmax, err)
											//                             If got really bad answer, report and
											//                             return.
											if fatal {
												goto label1120
											}
										}

									}

								}

							label170:
							}

						label180:
						}

					}

				label1100:
				}

			}

			//     Report result.
			if errmax < thresh {
				passL2(sname, nc, 17496, t)
			} else {
				t.Fail()
				fmt.Printf(" %6s COMPLETED THE COMPUTATIONAL TESTS (%6d CALLS)\n ******* BUT WITH MAXIMUM TEST RATIO%8.2f - SUSPECT *******\n", sname, nc, errmax)
			}
			continue

		label1120:
			;
			t.Fail()
			fmt.Printf(" ******* %6s FAILED ON CALL NUMBER:\n", sname)
			fmt.Printf(" %6d: %6s('%c','%c',%3d,%3d,%3d,%4.1f, A,%3d, B,%3d,%4.1f, C,%3d).\n", nc, sname, transa.Byte(), transb.Byte(), m, n, k, alpha, lda, ldb, beta, ldc)

		} else if sname == "Zhemm" || sname == "Zsymm" {
			conj := string((sname)[1:3]) == "he"
			if conj {
				zchkeLevel3("Zhemm")
			} else {
				zchkeLevel3("Zsymm")
			}

			nargs = 12
			nc = 0
			reset = true
			errmax = rzero

			for im = 1; im <= nidim; im++ {
				m = idim[im-1]

				for in = 1; in <= nidim; in++ {
					n = idim[in-1]
					//           Set LDC to 1 more than minimum value if room.
					ldc = m
					if ldc < nmax {
						ldc = ldc + 1
					}
					//           Skip tests if not enough room.
					if ldc > nmax {
						goto label290
					}
					lcc = ldc * n
					null = n <= 0 || m <= 0
					//           Set LDB to 1 more than minimum value if room.
					ldb = m
					if ldb < nmax {
						ldb = ldb + 1
					}
					//           Skip tests if not enough room.
					if ldb > nmax {
						goto label290
					}
					lbb = ldb * n

					//           Generate the matrix B.
					zmakeL3([]byte("ge"), Full, NonUnit, m, n, ab.Off(0, nmax+1-1), nmax, bb, ldb, &reset, zero)

					for ics = 1; ics <= 2; ics++ {
						side = ichs[ics-1]
						left = side == Left

						if left {
							na = m
						} else {
							na = n
						}
						//              Set LDA to 1 more than minimum value if room.
						lda = na
						if lda < nmax {
							lda = lda + 1
						}
						//              Skip tests if not enough room.
						if lda > nmax {
							goto label280
						}
						laa = lda * na

						for icu = 1; icu <= 2; icu++ {
							uplo = ichu[icu-1]

							//                 Generate the hermitian or symmetric matrix A.
							zmakeL3([]byte((sname)[1:3]), uplo, NonUnit, na, na, ab, nmax, aa, lda, &reset, zero)

							for ia = 1; ia <= nalf; ia++ {
								alpha = alf.Get(ia - 1)

								for ib = 1; ib <= nbet; ib++ {
									beta = bet.Get(ib - 1)

									//                       Generate the matrix C.
									zmakeL3([]byte("ge"), Full, mat.NonUnit, m, n, c, nmax, cc, ldc, &reset, zero)

									nc = nc + 1

									//                       Save every datum before calling the
									//                       subroutine.
									sides = side
									uplos = uplo
									ms = m
									ns = n
									als = alpha
									as = aa.DeepCopy()
									ldas = lda
									bs = bb.DeepCopy()
									ldbs = ldb
									bls = beta
									cs = cc.DeepCopy()
									ldcs = ldc

									//                       Call the subroutine.
									if conj {
										err2 = Zhemm(side, uplo, m, n, alpha, aa.CMatrix(lda, opts), lda, bb.CMatrix(ldb, opts), ldb, beta, cc.CMatrix(ldc, opts), ldc)
									} else {
										err2 = Zsymm(side, uplo, m, n, alpha, aa.CMatrix(lda, opts), lda, bb.CMatrix(ldb, opts), ldb, beta, cc.CMatrix(ldc, opts), ldc)
									}

									//                       Check if error-exit was taken incorrectly.
									if !(*ok) {
										t.Fail()
										fmt.Printf(" ******* FATAL ERROR - ERROR-EXIT TAKEN ON VALID CALL *******\n")
										fatal = true
										goto label2110
									}

									//                       See what data changed inside subroutines.
									isame[0] = sides == side
									isame[1] = uplos == uplo
									isame[2] = ms == m
									isame[3] = ns == n
									isame[4] = als == alpha
									isame[5] = lze(as, aa, laa)
									isame[6] = ldas == lda
									isame[7] = lze(bs, bb, lbb)
									isame[8] = ldbs == ldb
									isame[9] = bls == beta
									if null {
										isame[10] = lze(cs, cc, lcc)
									} else {
										isame[10] = lzeres([]byte("ge"), Full, m, n, cs.CMatrix(ldc, opts), cc.CMatrix(ldc, opts), ldc)
									}
									isame[11] = ldcs == ldc

									//                       If data was incorrectly changed, report and
									//                       return.
									same = true
									for i = 1; i <= nargs; i++ {
										same = same && isame[i-1]
										if !isame[i-1] {
											t.Fail()
											fmt.Printf(" ******* FATAL ERROR - PARAMETER NUMBER %2d WAS CHANGED INCORRECTLY *******\n", i)
										}
									}
									if !same {
										fatal = true
										goto label2110
									}

									if !null {
										//                          Check the result.
										if left {
											zmmch(NoTrans, NoTrans, m, n, m, alpha, ab, nmax, ab.Off(0, nmax+1-1), nmax, beta, c, nmax, ct, g, cc.CMatrix(ldc, opts), ldc, eps, &err, &fatal, true, t)
										} else {
											zmmch(NoTrans, NoTrans, m, n, n, alpha, ab.Off(0, nmax+1-1), nmax, ab, nmax, beta, c, nmax, ct, g, cc.CMatrix(ldc, opts), ldc, eps, &err, &fatal, true, t)
										}
										errmax = math.Max(errmax, err)
										//                          If got really bad answer, report and
										//                          return.
										if fatal {
											goto label2110
										}
									}

								}

							}

						}

					label280:
					}

				label290:
				}

			}

			//     Report result.
			if errmax < thresh {
				passL2(sname, nc, 1296, t)
			} else {
				t.Fail()
				fmt.Printf(" %6s COMPLETED THE COMPUTATIONAL TESTS (%6d CALLS)\n ******* BUT WITH MAXIMUM TEST RATIO%8.2f - SUSPECT *******\n", sname, nc, errmax)
			}
			continue

		label2110:
			;
			t.Fail()
			fmt.Printf(" ******* %6s FAILED ON CALL NUMBER:\n", sname)
			fmt.Printf(" %6d: %6s('%c','%c',%3d,%3d,%4.1f, A,%3d, B,%3d,%4.1f, C,%3d)    .\n", nc, sname, side.Byte(), uplo.Byte(), m, n, alpha, lda, ldb, beta, ldc)

		} else if sname == "Ztrmm" || sname == "Ztrsm" {
			if sname[3:] == "mm" {
				zchkeLevel3("Ztrmm")
			} else {
				zchkeLevel3("Ztrsm")
			}

			a := ab
			b := ab.Off(0, nmax+1-1)
			nargs = 11
			nc = 0
			reset = true
			errmax = rzero
			//     Set up zero matrix for ZMMCH.
			for j = 1; j <= nmax; j++ {
				for i = 1; i <= nmax; i++ {
					c.Set(i-1, j-1, zero)
				}
			}

			for im = 1; im <= nidim; im++ {
				m = idim[im-1]

				for in = 1; in <= nidim; in++ {
					n = idim[in-1]
					//           Set LDB to 1 more than minimum value if room.
					ldb = m
					if ldb < nmax {
						ldb = ldb + 1
					}
					//           Skip tests if not enough room.
					if ldb > nmax {
						goto label130
					}
					lbb = ldb * n
					null = m <= 0 || n <= 0

					for ics = 1; ics <= 2; ics++ {
						side = ichs[ics-1]
						left = side == Left
						if left {
							na = m
						} else {
							na = n
						}
						//              Set LDA to 1 more than minimum value if room.
						lda = na
						if lda < nmax {
							lda = lda + 1
						}
						//              Skip tests if not enough room.
						if lda > nmax {
							goto label130
						}
						laa = lda * na

						for icu = 1; icu <= 2; icu++ {
							uplo = ichu[icu-1]

							for ict = 1; ict <= 3; ict++ {
								transa = icht[ict-1]

								for icd = 1; icd <= 2; icd++ {
									diag = ichd[icd-1]

									for ia = 1; ia <= nalf; ia++ {
										alpha = alf.Get(ia - 1)

										//                          Generate the matrix A.
										zmakeL3([]byte("tr"), uplo, diag, na, na, a, nmax, aa, lda, &reset, zero)

										//                          Generate the matrix B.
										zmakeL3([]byte("ge"), Full, NonUnit, m, n, b, nmax, bb, ldb, &reset, zero)

										nc = nc + 1

										//                          Save every datum before calling the
										//                          subroutine.
										sides = side
										uplos = uplo
										tranas = transa
										diags = diag
										ms = m
										ns = n
										als = alpha
										as = aa.DeepCopy()
										ldas = lda
										bs = bb.DeepCopy()
										ldbs = ldb

										//                          Call the subroutine.
										if string((sname)[3:5]) == "mm" {
											err2 = Ztrmm(side, uplo, transa, diag, m, n, alpha, aa.CMatrix(lda, opts), lda, bb.CMatrix(ldb, opts), ldb)
										} else if string((sname)[3:5]) == "sm" {
											err2 = Ztrsm(side, uplo, transa, diag, m, n, alpha, aa.CMatrix(lda, opts), lda, bb.CMatrix(ldb, opts), ldb)
										}

										//                          Check if error-exit was taken incorrectly.
										if !(*ok) {
											t.Fail()
											fmt.Printf(" ******* FATAL ERROR - ERROR-EXIT TAKEN ON VALID CALL *******\n")
											fatal = true
											goto label150
										}

										//                          See what data changed inside subroutines.
										isame[0] = sides == side
										isame[1] = uplos == uplo
										isame[2] = tranas == transa
										isame[3] = diags == diag
										isame[4] = ms == m
										isame[5] = ns == n
										isame[6] = als == alpha
										isame[7] = lze(as, aa, laa)
										isame[8] = ldas == lda
										if null {
											isame[9] = lze(bs, bb, lbb)
										} else {
											isame[9] = lzeres([]byte("ge"), Full, m, n, bs.CMatrix(ldb, opts), bb.CMatrix(ldb, opts), ldb)
										}
										isame[10] = ldbs == ldb

										//                          If data was incorrectly changed, report and
										//                          return.
										same = true
										for i = 1; i <= nargs; i++ {
											same = same && isame[i-1]
											if !isame[i-1] {
												t.Fail()
												fmt.Printf(" ******* FATAL ERROR - PARAMETER NUMBER %2d WAS CHANGED INCORRECTLY *******\n", i)
											}
										}
										if !same {
											fatal = true
											goto label150
										}

										if !null {
											if string((sname)[3:5]) == "mm" {
												//                                Check the result.
												if left {
													zmmch(transa, NoTrans, m, n, m, alpha, a, nmax, b, nmax, zero, c, nmax, ct, g, bb.CMatrix(ldb, opts), ldb, eps, &err, &fatal, true, t)
												} else {
													zmmch(NoTrans, transa, m, n, n, alpha, b, nmax, ab, nmax, zero, c, nmax, ct, g, bb.CMatrix(ldb, opts), ldb, eps, &err, &fatal, true, t)
												}
											} else if string((sname)[3:5]) == "sm" {
												//                                Compute approximation to original
												//                                matrix.
												for j = 1; j <= n; j++ {
													for i = 1; i <= m; i++ {
														c.Set(i-1, j-1, bb.Get(i+(j-1)*ldb-1))
														bb.Set(i+(j-1)*ldb-1, alpha*b.Get(i-1, j-1))
													}
												}

												if left {
													zmmch(transa, NoTrans, m, n, m, one, a, nmax, c, nmax, zero, b, nmax, ct, g, bb.CMatrix(ldb, opts), ldb, eps, &err, &fatal, false, t)
												} else {
													zmmch(NoTrans, transa, m, n, n, one, c, nmax, a, nmax, zero, b, nmax, ct, g, bb.CMatrix(ldb, opts), ldb, eps, &err, &fatal, false, t)
												}
											}
											errmax = math.Max(errmax, err)
											//                             If got really bad answer, report and
											//                             return.
											if fatal {
												goto label150
											}
										}

									}

								}

							}

						}

					}

				label130:
				}

			}

			//     Report result.
			if errmax < thresh {
				passL2(sname, nc, 2592, t)
			} else {
				t.Fail()
				fmt.Printf(" %6s COMPLETED THE COMPUTATIONAL TESTS (%6d CALLS)\n ******* BUT WITH MAXIMUM TEST RATIO%8.2f - SUSPECT *******\n", sname, nc, errmax)
			}
			continue

		label150:
			;
			t.Fail()
			fmt.Printf(" ******* %6s FAILED ON CALL NUMBER:\n", sname)
			fmt.Printf(" %6d: %6s('%c','%c','%c','%c',%3d,%3d,%4.1f, A,%3d, B,%3d)               .\n", nc, sname, side.Byte(), uplo.Byte(), transa.Byte(), diag.Byte(), m, n, alpha, lda, ldb)

		} else if sname == "Zherk" || sname == "Zsyrk" {
			a := ab
			conj := string((sname)[1:3]) == "he"
			icht := []mat.MatTrans{NoTrans, ConjTrans}

			if conj {
				zchkeLevel3("Zherk")
			} else {
				zchkeLevel3("Zsyrk")
			}

			nargs = 10
			nc = 0
			reset = true
			errmax = rzero

			for in = 1; in <= nidim; in++ {
				n = idim[in-1]
				//        Set LDC to 1 more than minimum value if room.
				ldc = n
				if ldc < nmax {
					ldc = ldc + 1
				}
				//        Skip tests if not enough room.
				if ldc > nmax {
					goto label4100
				}
				lcc = ldc * n

				for ik = 1; ik <= nidim; ik++ {
					k = idim[ik-1]

					for ict = 1; ict <= 2; ict++ {
						trans = icht[ict-1]
						tran := trans == ConjTrans
						if tran && !conj {
							trans = Trans
						}
						if tran {
							ma = k
							na = n
						} else {
							ma = n
							na = k
						}
						//              Set LDA to 1 more than minimum value if room.
						lda = ma
						if lda < nmax {
							lda = lda + 1
						}
						//              Skip tests if not enough room.
						if lda > nmax {
							goto label480
						}
						laa = lda * na

						//              Generate the matrix A.
						zmakeL3([]byte("ge"), Full, NonUnit, ma, na, a, nmax, aa, lda, &reset, zero)

						for icu = 1; icu <= 2; icu++ {
							uplo = ichu[icu-1]
							upper = uplo == Upper

							for ia = 1; ia <= nalf; ia++ {
								alpha = alf.Get(ia - 1)
								if conj {
									ralpha = real(alpha)
									alpha = complex(ralpha, rzero)
								}

								for ib = 1; ib <= nbet; ib++ {
									beta = bet.Get(ib - 1)
									if conj {
										rbeta = real(beta)
										beta = complex(rbeta, rzero)
									}
									null = n <= 0
									if conj {
										null = null || ((k <= 0 || ralpha == rzero) && rbeta == rone)
									}

									//                       Generate the matrix C.
									zmakeL3([]byte((sname)[1:3]), uplo, mat.NonUnit, n, n, c, nmax, cc, ldc, &reset, zero)

									nc = nc + 1

									//                       Save every datum before calling the subroutine.
									uplos = uplo
									transs = trans
									ns = n
									ks = k
									if conj {
										rals = ralpha
									} else {
										als = alpha
									}
									as = aa.DeepCopy()
									ldas = lda
									if conj {
										rbets = rbeta
									} else {
										bets = beta
									}
									cs = cc.DeepCopy()
									ldcs = ldc

									//                       Call the subroutine.
									if conj {
										err2 = Zherk(uplo, trans, n, k, ralpha, aa.CMatrix(lda, opts), lda, rbeta, cc.CMatrix(ldc, opts), ldc)
									} else {
										err2 = Zsyrk(uplo, trans, n, k, alpha, aa.CMatrix(lda, opts), lda, beta, cc.CMatrix(ldc, opts), ldc)
									}

									//                       Check if error-exit was taken incorrectly.
									if !(*ok) {
										t.Fail()
										fmt.Printf(" ******* FATAL ERROR - ERROR-EXIT TAKEN ON VALID CALL *******\n")
										fatal = true
										goto label4120
									}

									//                       See what data changed inside subroutines.
									isame[0] = uplos == uplo
									isame[1] = transs == trans
									isame[2] = ns == n
									isame[3] = ks == k
									if conj {
										isame[4] = rals == ralpha
									} else {
										isame[4] = als == alpha
									}
									isame[5] = lze(as, aa, laa)
									isame[6] = ldas == lda
									if conj {
										isame[7] = rbets == rbeta
									} else {
										isame[7] = bets == beta
									}
									if null {
										isame[8] = lze(cs, cc, lcc)
									} else {
										isame[8] = lzeres([]byte((sname)[1:3]), uplo, n, n, cs.CMatrix(ldc, opts), cc.CMatrix(ldc, opts), ldc)
									}
									isame[9] = ldcs == ldc

									//                       If data was incorrectly changed, report and
									//                       return.
									same = true
									for i = 1; i <= nargs; i++ {
										same = same && isame[i-1]
										if !isame[i-1] {
											t.Fail()
											fmt.Printf(" ******* FATAL ERROR - PARAMETER NUMBER %2d WAS CHANGED INCORRECTLY *******\n", i)
										}
									}
									if !same {
										fatal = true
										goto label4120
									}

									if !null {
										//                          Check the result column by column.
										if conj {
											transt = ConjTrans
										} else {
											transt = Trans
										}
										jc = 1
										for j = 1; j <= n; j++ {
											if upper {
												jj = 1
												lj = j
											} else {
												jj = j
												lj = n - j + 1
											}
											if tran {
												zmmch(transt, NoTrans, lj, 1, k, alpha, a.Off(0, jj-1), nmax, a.Off(0, j-1), nmax, beta, c.Off(jj-1, j-1), nmax, ct, g, cc.CMatrixOff(jc-1, ldc, opts), ldc, eps, &err, &fatal, true, t)
											} else {
												zmmch(NoTrans, transt, lj, 1, k, alpha, a.Off(jj-1, 0), nmax, a.Off(j-1, 0), nmax, beta, c.Off(jj-1, j-1), nmax, ct, g, cc.CMatrixOff(jc-1, ldc, opts), ldc, eps, &err, &fatal, true, t)
											}
											if upper {
												jc = jc + ldc
											} else {
												jc = jc + ldc + 1
											}
											errmax = math.Max(errmax, err)
											//                             If got really bad answer, report and
											//                             return.
											if fatal {
												goto label4110
											}
										}
									}

								}

							}

						}

					label480:
					}

				}

			label4100:
			}

			//     Report result.
			if errmax < thresh {
				passL2(sname, nc, 1296, t)
			} else {
				t.Fail()
				fmt.Printf(" %6s COMPLETED THE COMPUTATIONAL TESTS (%6d CALLS)\n ******* BUT WITH MAXIMUM TEST RATIO%8.2f - SUSPECT *******\n", sname, nc, errmax)
			}
			continue

		label4110:
			;
			if n > 1 {
				fmt.Printf("      THESE ARE THE RESULTS FOR COLUMN %3d\n", j)
			}

		label4120:
			;
			t.Fail()
			fmt.Printf(" ******* %6s FAILED ON CALL NUMBER:\n", sname)
			if conj {
				fmt.Printf(" %6d: %6s('%c','%c',%3d,%3d,%4.1f, A,%3d,%4.1f, C,%3d)                         .\n", nc, sname, uplo.Byte(), trans.Byte(), n, k, ralpha, lda, rbeta, ldc)
			} else {
				fmt.Printf(" %6d: %6s('%c','%c',%3d,%3d,%4.1f , A,%3d,%4.1f, C,%3d)          .\n", nc, sname, uplo.Byte(), trans.Byte(), n, k, alpha, lda, beta, ldc)
			}

		} else if sname == "Zher2k" || sname == "Zsyr2k" {
			_ab := ab.CVectorIdx(0)
			conj := string((sname)[1:3]) == "he"
			icht := []mat.MatTrans{NoTrans, ConjTrans}

			if conj {
				zchkeLevel3("Zher2k")
			} else {
				zchkeLevel3("Zsyr2k")
			}

			nargs = 12
			nc = 0
			reset = true
			errmax = rzero

			for in = 1; in <= nidim; in++ {
				n = idim[in-1]
				//        Set LDC to 1 more than minimum value if room.
				ldc = n
				if ldc < nmax {
					ldc = ldc + 1
				}
				//        Skip tests if not enough room.
				if ldc > nmax {
					goto label5130
				}
				lcc = ldc * n

				for ik = 1; ik <= nidim; ik++ {
					k = idim[ik-1]

					for ict = 1; ict <= 2; ict++ {
						trans = icht[ict-1]
						tran := trans == ConjTrans
						if tran && !conj {
							trans = Trans
						}
						if tran {
							ma = k
							na = n
						} else {
							ma = n
							na = k
						}
						//              Set LDA to 1 more than minimum value if room.
						lda = ma
						if lda < nmax {
							lda = lda + 1
						}
						//              Skip tests if not enough room.
						if lda > nmax {
							goto label5110
						}
						laa = lda * na

						//              Generate the matrix A.
						if tran {
							zmakeL3([]byte("ge"), Full, NonUnit, ma, na, _ab.CMatrix(2*nmax, opts), 2*nmax, aa, lda, &reset, zero)
						} else {
							zmakeL3([]byte("ge"), Full, NonUnit, ma, na, _ab.CMatrix(nmax, opts), nmax, aa, lda, &reset, zero)
						}

						//              Generate the matrix B.
						ldb = lda
						lbb = laa
						if tran {
							zmakeL3([]byte("ge"), Full, NonUnit, ma, na, _ab.CMatrixOff(k+1-1, 2*nmax, opts), 2*nmax, bb, ldb, &reset, zero)
						} else {
							zmakeL3([]byte("ge"), Full, NonUnit, ma, na, _ab.CMatrixOff(k*nmax+1-1, nmax, opts), nmax, bb, ldb, &reset, zero)
						}

						for icu = 1; icu <= 2; icu++ {
							uplo = ichu[icu-1]
							upper = uplo == Upper

							for ia = 1; ia <= nalf; ia++ {
								alpha = alf.Get(ia - 1)

								for ib = 1; ib <= nbet; ib++ {
									beta = bet.Get(ib - 1)
									if conj {
										rbeta = real(beta)
										beta = complex(rbeta, rzero)
									}
									null = n <= 0
									if conj {
										null = null || ((k <= 0 || alpha == zero) && rbeta == rone)
									}

									//                       Generate the matrix C.
									zmakeL3([]byte((sname)[1:3]), uplo, NonUnit, n, n, c, nmax, cc, ldc, &reset, zero)

									nc = nc + 1

									//                       Save every datum before calling the subroutine.
									uplos = uplo
									transs = trans
									ns = n
									ks = k
									als = alpha
									as = aa.DeepCopy()
									ldas = lda
									bs = bb.DeepCopy()
									ldbs = ldb
									if conj {
										rbets = rbeta
									} else {
										bets = beta
									}
									cs = cc.DeepCopy()
									ldcs = ldc

									//                       Call the subroutine.
									if conj {
										err2 = Zher2k(uplo, trans, n, k, alpha, aa.CMatrix(lda, opts), lda, bb.CMatrix(ldb, opts), ldb, rbeta, cc.CMatrix(ldc, opts), ldc)
									} else {
										err2 = Zsyr2k(uplo, trans, n, k, alpha, aa.CMatrix(lda, opts), lda, bb.CMatrix(ldb, opts), ldb, beta, cc.CMatrix(ldc, opts), ldc)
									}

									//                       Check if error-exit was taken incorrectly.
									if !(*ok) {
										t.Fail()
										fmt.Printf(" ******* FATAL ERROR - ERROR-EXIT TAKEN ON VALID CALL *******\n")
										fatal = true
										goto label5150
									}

									//                       See what data changed inside subroutines.
									isame[0] = uplos == uplo
									isame[1] = transs == trans
									isame[2] = ns == n
									isame[3] = ks == k
									isame[4] = als == alpha
									isame[5] = lze(as, aa, laa)
									isame[6] = ldas == lda
									isame[7] = lze(bs, bb, lbb)
									isame[8] = ldbs == ldb
									if conj {
										isame[9] = rbets == rbeta
									} else {
										isame[9] = bets == beta
									}
									if null {
										isame[10] = lze(cs, cc, lcc)
									} else {
										isame[10] = lzeres([]byte("he"), uplo, n, n, cs.CMatrix(ldc, opts), cc.CMatrix(ldc, opts), ldc)
									}
									isame[11] = ldcs == ldc

									//                       If data was incorrectly changed, report and
									//                       return.
									same = true
									for i = 1; i <= nargs; i++ {
										same = same && isame[i-1]
										if !isame[i-1] {
											t.Fail()
											fmt.Printf(" ******* FATAL ERROR - PARAMETER NUMBER %2d WAS CHANGED INCORRECTLY *******\n", i)
										}
									}
									if !same {
										fatal = true
										goto label5150
									}

									if !null {
										//                          Check the result column by column.
										if conj {
											transt = ConjTrans
										} else {
											transt = Trans
										}
										jjab = 1
										jc = 1
										for j = 1; j <= n; j++ {
											if upper {
												jj = 1
												lj = j
											} else {
												jj = j
												lj = n - j + 1
											}
											if tran {
												for i = 1; i <= k; i++ {
													w.Set(i-1, alpha*ab.GetIdx((j-1)*2*nmax+k+i-1))
													if conj {
														w.Set(k+i-1, cmplx.Conj(alpha)*ab.GetIdx((j-1)*2*nmax+i-1))
													} else {
														w.Set(k+i-1, alpha*ab.GetIdx((j-1)*2*nmax+i-1))
													}
												}
												zmmch(transt, NoTrans, lj, 1, 2*k, one, _ab.CMatrixOff(jjab-1, 2*nmax, opts), 2*nmax, w.CMatrix(2*nmax, opts), 2*nmax, beta, c.Off(jj-1, j-1), nmax, ct, g, cc.CMatrixOff(jc-1, ldc, opts), ldc, eps, &err, &fatal, true, t)
											} else {
												for i = 1; i <= k; i++ {
													if conj {
														w.Set(i-1, alpha*ab.GetConjIdx((k+i-1)*nmax+j-1))
														w.Set(k+i-1, cmplx.Conj(alpha*ab.GetIdx((i-1)*nmax+j-1)))
													} else {
														w.Set(i-1, alpha*ab.GetIdx((k+i-1)*nmax+j-1))
														w.Set(k+i-1, alpha*ab.GetIdx((i-1)*nmax+j-1))
													}
												}
												zmmch(NoTrans, NoTrans, lj, 1, 2*k, one, _ab.CMatrixOff(jj-1, nmax, opts), nmax, w.CMatrix(2*nmax, opts), 2*nmax, beta, c.Off(jj-1, j-1), nmax, ct, g, cc.CMatrixOff(jc-1, ldc, opts), ldc, eps, &err, &fatal, true, t)
											}
											if upper {
												jc = jc + ldc
											} else {
												jc = jc + ldc + 1
												if tran {
													jjab = jjab + 2*nmax
												}
											}
											errmax = math.Max(errmax, err)
											//                             If got really bad answer, report and
											//                             return.
											if fatal {
												goto label5140
											}
										}
									}

								}

							}

						}

					label5110:
					}

				}

			label5130:
			}

			//     Report result.
			if errmax < thresh {
				passL2(sname, nc, 1296, t)
			} else {
				t.Fail()
				fmt.Printf(" %6s COMPLETED THE COMPUTATIONAL TESTS (%6d CALLS)\n ******* BUT WITH MAXIMUM TEST RATIO%8.2f - SUSPECT *******\n", sname, nc, errmax)
			}
			continue

		label5140:
			;
			if n > 1 {
				fmt.Printf("      THESE ARE THE RESULTS FOR COLUMN %3d\n", j)
			}
			//
		label5150:
			;
			t.Fail()
			fmt.Printf(" ******* %6s FAILED ON CALL NUMBER:\n", sname)
			if conj {
				fmt.Printf(" %6d: %6s('%c','%c',%3d,%3d,%4.1f, A,%3d, B,%3d,%4.1f, C,%3d)           .\n", nc, sname, uplo.Byte(), trans.Byte(), n, k, alpha, lda, ldb, rbeta, ldc)
			} else {
				fmt.Printf(" %6d: %6s('%c','%c',%3d,%3d,%4.1f, A,%3d, B,%3d,%4.1f, C,%3d)    .\n", nc, sname, uplo.Byte(), trans.Byte(), n, k, alpha, lda, ldb, beta, ldc)
			}

		}
	}
}

func zmakeL3(_type []byte, uplo mat.MatUplo, diag mat.MatDiag, m, n int, a *mat.CMatrix, nmax int, aa *mat.CVector, lda int, reset *bool, transl complex128) {
	var gen, her, lower, sym, tri, unit, upper bool
	var one, rogue, zero complex128
	var rrogue, rzero float64
	var i, ibeg, iend, j, jj int

	zero = (0.0 + 0.0*1i)
	one = (1.0 + 0.0*1i)
	rogue = (-1.0e10 + 1.0e10*1i)
	rzero = 0.0
	rrogue = -1.0e10

	gen = string(_type) == "GE" || string(_type) == "ge"
	her = string(_type) == "HE" || string(_type) == "he"
	sym = string(_type) == "SY" || string(_type) == "sy"
	tri = string(_type) == "TR" || string(_type) == "tr"
	upper = (her || sym || tri) && uplo == Upper
	lower = (her || sym || tri) && uplo == Lower
	unit = tri && diag == Unit
	//
	//     Generate data in array A.
	//
	for j = 1; j <= n; j++ {
		for i = 1; i <= m; i++ {
			if gen || (upper && i <= j) || (lower && i >= j) {
				a.Set(i-1, j-1, zbeg(reset)+transl)
				if i != j {
					//                 Set some elements to zero
					if n > 3 && j == n/2 {
						a.Set(i-1, j-1, zero)
					}
					if her {
						a.Set(j-1, i-1, a.GetConj(i-1, j-1))
					} else if sym {
						a.Set(j-1, i-1, a.Get(i-1, j-1))
					} else if tri {
						a.Set(j-1, i-1, zero)
					}
				}
			}
		}
		if her {
			a.Set(j-1, j-1, complex(real(a.Get(j-1, j-1)), rzero))
		}
		if tri {
			a.Set(j-1, j-1, a.Get(j-1, j-1)+one)
		}
		if unit {
			a.Set(j-1, j-1, one)
		}
	}

	//     Store elements in array AS in data structure required by routine.
	if string(_type) == "GE" || string(_type) == "ge" {
		for j = 1; j <= n; j++ {
			for i = 1; i <= m; i++ {
				aa.Set(i+(j-1)*lda-1, a.Get(i-1, j-1))
			}
			for i = m + 1; i <= lda; i++ {
				aa.Set(i+(j-1)*lda-1, rogue)
			}
		}
	} else if string(_type) == "HE" || string(_type) == "SY" || string(_type) == "TR" || string(_type) == "he" || string(_type) == "sy" || string(_type) == "tr" {
		for j = 1; j <= n; j++ {
			if upper {
				ibeg = 1
				if unit {
					iend = j - 1
				} else {
					iend = j
				}
			} else {
				if unit {
					ibeg = j + 1
				} else {
					ibeg = j
				}
				iend = n
			}
			for i = 1; i <= ibeg-1; i++ {
				aa.Set(i+(j-1)*lda-1, rogue)
			}
			for i = ibeg; i <= iend; i++ {
				aa.Set(i+(j-1)*lda-1, a.Get(i-1, j-1))
			}
			for i = iend + 1; i <= lda; i++ {
				aa.Set(i+(j-1)*lda-1, rogue)
			}
			if her {
				jj = j + (j-1)*lda
				aa.Set(jj-1, complex(real(aa.Get(jj-1)), rrogue))
			}
		}
	}
}

func zmmch(transa, transb mat.MatTrans, m, n, kk int, alpha complex128, a *mat.CMatrix, lda int, b *mat.CMatrix, ldb int, beta complex128, c *mat.CMatrix, ldc int, ct *mat.CVector, g *mat.Vector, cc *mat.CMatrix, ldcc int, eps float64, err *float64, fatal *bool, mv bool, t *testing.T) {
	var ctrana, ctranb, trana, tranb bool
	var zero complex128
	var erri, rone, rzero float64
	var i, j, k int

	zero = (0.0 + 0.0*1i)
	rzero = 0.0
	rone = 1.0

	Abs1 := func(cl complex128) float64 { return math.Abs(real(cl)) + math.Abs(imag(cl)) }

	trana = transa == Trans || transa == ConjTrans
	tranb = transb == Trans || transb == ConjTrans
	ctrana = transa == ConjTrans
	ctranb = transb == ConjTrans

	//     Compute expected result, one column at a time, in CT using data
	//     in A, B and C.
	//     Compute gauges in G.
	for j = 1; j <= n; j++ {

		for i = 1; i <= m; i++ {
			ct.Set(i-1, zero)
			g.Set(i-1, rzero)
		}
		if !trana && !tranb {
			for k = 1; k <= kk; k++ {
				for i = 1; i <= m; i++ {
					ct.Set(i-1, ct.Get(i-1)+a.Get(i-1, k-1)*b.Get(k-1, j-1))
					g.Set(i-1, g.Get(i-1)+Abs1(a.Get(i-1, k-1))*Abs1(b.Get(k-1, j-1)))
				}
			}
		} else if trana && !tranb {
			if ctrana {
				for k = 1; k <= kk; k++ {
					for i = 1; i <= m; i++ {
						ct.Set(i-1, ct.Get(i-1)+a.GetConj(k-1, i-1)*b.Get(k-1, j-1))
						g.Set(i-1, g.Get(i-1)+Abs1(a.Get(k-1, i-1))*Abs1(b.Get(k-1, j-1)))
					}
				}
			} else {
				for k = 1; k <= kk; k++ {
					for i = 1; i <= m; i++ {
						ct.Set(i-1, ct.Get(i-1)+a.Get(k-1, i-1)*b.Get(k-1, j-1))
						g.Set(i-1, g.Get(i-1)+Abs1(a.Get(k-1, i-1))*Abs1(b.Get(k-1, j-1)))
					}
				}
			}
		} else if !trana && tranb {
			if ctranb {
				for k = 1; k <= kk; k++ {
					for i = 1; i <= m; i++ {
						ct.Set(i-1, ct.Get(i-1)+a.Get(i-1, k-1)*b.GetConj(j-1, k-1))
						g.Set(i-1, g.Get(i-1)+Abs1(a.Get(i-1, k-1))*Abs1(b.Get(j-1, k-1)))
					}
				}
			} else {
				for k = 1; k <= kk; k++ {
					for i = 1; i <= m; i++ {
						ct.Set(i-1, ct.Get(i-1)+a.Get(i-1, k-1)*b.Get(j-1, k-1))
						g.Set(i-1, g.Get(i-1)+Abs1(a.Get(i-1, k-1))*Abs1(b.Get(j-1, k-1)))
					}
				}
			}
		} else if trana && tranb {
			if ctrana {
				if ctranb {
					for k = 1; k <= kk; k++ {
						for i = 1; i <= m; i++ {
							ct.Set(i-1, ct.Get(i-1)+a.GetConj(k-1, i-1)*b.GetConj(j-1, k-1))
							g.Set(i-1, g.Get(i-1)+Abs1(a.Get(k-1, i-1))*Abs1(b.Get(j-1, k-1)))
						}
					}
				} else {
					for k = 1; k <= kk; k++ {
						for i = 1; i <= m; i++ {
							ct.Set(i-1, ct.Get(i-1)+a.GetConj(k-1, i-1)*b.Get(j-1, k-1))
							g.Set(i-1, g.Get(i-1)+Abs1(a.Get(k-1, i-1))*Abs1(b.Get(j-1, k-1)))
						}
					}
				}
			} else {
				if ctranb {
					for k = 1; k <= kk; k++ {
						for i = 1; i <= m; i++ {
							ct.Set(i-1, ct.Get(i-1)+a.Get(k-1, i-1)*b.GetConj(j-1, k-1))
							g.Set(i-1, g.Get(i-1)+Abs1(a.Get(k-1, i-1))*Abs1(b.Get(j-1, k-1)))
						}
					}
				} else {
					for k = 1; k <= kk; k++ {
						for i = 1; i <= m; i++ {
							ct.Set(i-1, ct.Get(i-1)+a.Get(k-1, i-1)*b.Get(j-1, k-1))
							g.Set(i-1, g.Get(i-1)+Abs1(a.Get(k-1, i-1))*Abs1(b.Get(j-1, k-1)))
						}
					}
				}
			}
		}
		for i = 1; i <= m; i++ {
			ct.Set(i-1, alpha*ct.Get(i-1)+beta*c.Get(i-1, j-1))
			g.Set(i-1, Abs1(alpha)*g.Get(i-1)+Abs1(beta)*Abs1(c.Get(i-1, j-1)))
		}

		//        Compute the error ratio for this result.
		(*err) = rzero
		for i = 1; i <= m; i++ {
			erri = Abs1(ct.Get(i-1)-cc.Get(i-1, j-1)) / eps
			if g.Get(i-1) != rzero {
				erri = erri / g.Get(i-1)
			}
			(*err) = math.Max(*err, erri)
			if (*err)*math.Sqrt(eps) >= rone {
				goto label230
			}
		}

	}

	//     If the loop completes, all results are at least half accurate.
	return

	//     Report fatal error.
label230:
	;
	*fatal = true
	fmt.Printf(" ******* FATAL ERROR - COMPUTED RESULT IS LESS THAN HALF ACCURATE *******\n                       EXPECTED RESULT                    COMPUTED RESULT\n")
	for i = 1; i <= m; i++ {
		if mv {
			fmt.Printf(" %7d  (%15.6f,%15.6f)\n", i, ct.Get(i-1), cc.Get(i-1, j-1))
		} else {
			fmt.Printf(" %7d  (%15.6f,%15.6f)\n", i, cc.Get(i-1, j-1), ct.Get(i-1))
		}
	}
	if n > 1 {
		fmt.Printf("      THESE ARE THE RESULTS FOR COLUMN %3d\n", j)
	}

}

func zchkeLevel3(srnamt string) {
	var err error
	a := cmf(2, 1, opts)
	b := cmf(2, 1, opts)
	c := cmf(2, 1, opts)

	//
	//  Tests the error exits from the Level 3 Blas.
	//  Requires a special version of the error-handling routine XERBLA.
	//  A, B and C should not need to be defined.
	//
	//  Auxiliary routine for test program for Level 3 Blas.
	//
	//  -- Written on 8-February-1989.
	//     Jack Dongarra, Argonne National Laboratory.
	//     Iain Duff, AERE Harwell.
	//     Jeremy Du Croz, Numerical Algorithms Group Ltd.
	//     Sven Hammarling, Numerical Algorithms Group Ltd.
	//
	//  3-19-92:  Initialize ALPHA, BETA, RALPHA, and RBETA  (eca)
	//  3-19-92:  Fix argument 12 in calls to ZSYMM and ZHEMM
	//            with INFOT = 9  (eca)
	//  10-9-00:  Declared INTRINSIC DCMPLX (susan)
	//
	one := 1.0
	two := 2.0

	errt := &common.infoc.errt
	ok := &common.infoc.ok
	lerr := &common.infoc.lerr

	*ok = true
	*lerr = true

	alpha := complex(one, -one)
	beta := complex(two, -two)
	ralpha := one
	rbeta := two

	switch srnamt {
	case "Zgemm":
		*errt = fmt.Errorf("transa invalid: /")
		err = Zgemm('/', NoTrans, 0, 0, 0, alpha, a, 1, b, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("transa invalid: /")
		err = Zgemm('/', ConjTrans, 0, 0, 0, alpha, a, 1, b, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("transa invalid: /")
		err = Zgemm('/', Trans, 0, 0, 0, alpha, a, 1, b, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("transb invalid: /")
		err = Zgemm(NoTrans, '/', 0, 0, 0, alpha, a, 1, b, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("transb invalid: /")
		err = Zgemm(ConjTrans, '/', 0, 0, 0, alpha, a, 1, b, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("transb invalid: /")
		err = Zgemm(Trans, '/', 0, 0, 0, alpha, a, 1, b, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("m invalid: -1")
		err = Zgemm(NoTrans, NoTrans, -1, 0, 0, alpha, a, 1, b, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("m invalid: -1")
		err = Zgemm(NoTrans, ConjTrans, -1, 0, 0, alpha, a, 1, b, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("m invalid: -1")
		err = Zgemm(NoTrans, Trans, -1, 0, 0, alpha, a, 1, b, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("m invalid: -1")
		err = Zgemm(ConjTrans, NoTrans, -1, 0, 0, alpha, a, 1, b, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("m invalid: -1")
		err = Zgemm(ConjTrans, ConjTrans, -1, 0, 0, alpha, a, 1, b, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("m invalid: -1")
		err = Zgemm(ConjTrans, Trans, -1, 0, 0, alpha, a, 1, b, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("m invalid: -1")
		err = Zgemm(Trans, NoTrans, -1, 0, 0, alpha, a, 1, b, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("m invalid: -1")
		err = Zgemm(Trans, ConjTrans, -1, 0, 0, alpha, a, 1, b, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("m invalid: -1")
		err = Zgemm(Trans, Trans, -1, 0, 0, alpha, a, 1, b, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("n invalid: -1")
		err = Zgemm(NoTrans, NoTrans, 0, -1, 0, alpha, a, 1, b, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("n invalid: -1")
		err = Zgemm(NoTrans, ConjTrans, 0, -1, 0, alpha, a, 1, b, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("n invalid: -1")
		err = Zgemm(NoTrans, Trans, 0, -1, 0, alpha, a, 1, b, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("n invalid: -1")
		err = Zgemm(ConjTrans, NoTrans, 0, -1, 0, alpha, a, 1, b, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("n invalid: -1")
		err = Zgemm(ConjTrans, ConjTrans, 0, -1, 0, alpha, a, 1, b, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("n invalid: -1")
		err = Zgemm(ConjTrans, Trans, 0, -1, 0, alpha, a, 1, b, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("n invalid: -1")
		err = Zgemm(Trans, NoTrans, 0, -1, 0, alpha, a, 1, b, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("n invalid: -1")
		err = Zgemm(Trans, ConjTrans, 0, -1, 0, alpha, a, 1, b, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("n invalid: -1")
		err = Zgemm(Trans, Trans, 0, -1, 0, alpha, a, 1, b, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("k invalid: -1")
		err = Zgemm(NoTrans, NoTrans, 0, 0, -1, alpha, a, 1, b, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("k invalid: -1")
		err = Zgemm(NoTrans, ConjTrans, 0, 0, -1, alpha, a, 1, b, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("k invalid: -1")
		err = Zgemm(NoTrans, Trans, 0, 0, -1, alpha, a, 1, b, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("k invalid: -1")
		err = Zgemm(ConjTrans, NoTrans, 0, 0, -1, alpha, a, 1, b, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("k invalid: -1")
		err = Zgemm(ConjTrans, ConjTrans, 0, 0, -1, alpha, a, 1, b, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("k invalid: -1")
		err = Zgemm(ConjTrans, Trans, 0, 0, -1, alpha, a, 1, b, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("k invalid: -1")
		err = Zgemm(Trans, NoTrans, 0, 0, -1, alpha, a, 1, b, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("k invalid: -1")
		err = Zgemm(Trans, ConjTrans, 0, 0, -1, alpha, a, 1, b, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("k invalid: -1")
		err = Zgemm(Trans, Trans, 0, 0, -1, alpha, a, 1, b, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("lda invalid: 1")
		err = Zgemm(NoTrans, NoTrans, 2, 0, 0, alpha, a, 1, b, 1, beta, c, 2)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("lda invalid: 1")
		err = Zgemm(NoTrans, ConjTrans, 2, 0, 0, alpha, a, 1, b, 1, beta, c, 2)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("lda invalid: 1")
		err = Zgemm(NoTrans, Trans, 2, 0, 0, alpha, a, 1, b, 1, beta, c, 2)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("lda invalid: 1")
		err = Zgemm(ConjTrans, NoTrans, 0, 0, 2, alpha, a, 1, b, 2, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("lda invalid: 1")
		err = Zgemm(ConjTrans, ConjTrans, 0, 0, 2, alpha, a, 1, b, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("lda invalid: 1")
		err = Zgemm(ConjTrans, Trans, 0, 0, 2, alpha, a, 1, b, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("lda invalid: 1")
		err = Zgemm(Trans, NoTrans, 0, 0, 2, alpha, a, 1, b, 2, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("lda invalid: 1")
		err = Zgemm(Trans, ConjTrans, 0, 0, 2, alpha, a, 1, b, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("lda invalid: 1")
		err = Zgemm(Trans, Trans, 0, 0, 2, alpha, a, 1, b, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("ldb invalid: 1")
		err = Zgemm(NoTrans, NoTrans, 0, 0, 2, alpha, a, 1, b, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("ldb invalid: 1")
		err = Zgemm(ConjTrans, NoTrans, 0, 0, 2, alpha, a, 2, b, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("ldb invalid: 1")
		err = Zgemm(Trans, NoTrans, 0, 0, 2, alpha, a, 2, b, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("ldb invalid: 1")
		err = Zgemm(NoTrans, ConjTrans, 0, 2, 0, alpha, a, 1, b, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("ldb invalid: 1")
		err = Zgemm(ConjTrans, ConjTrans, 0, 2, 0, alpha, a, 1, b, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("ldb invalid: 1")
		err = Zgemm(Trans, ConjTrans, 0, 2, 0, alpha, a, 1, b, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("ldb invalid: 1")
		err = Zgemm(NoTrans, Trans, 0, 2, 0, alpha, a, 1, b, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("ldb invalid: 1")
		err = Zgemm(ConjTrans, Trans, 0, 2, 0, alpha, a, 1, b, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("ldb invalid: 1")
		err = Zgemm(Trans, Trans, 0, 2, 0, alpha, a, 1, b, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("ldc invalid: 1")
		err = Zgemm(NoTrans, NoTrans, 2, 0, 0, alpha, a, 2, b, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("ldc invalid: 1")
		err = Zgemm(NoTrans, ConjTrans, 2, 0, 0, alpha, a, 2, b, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("ldc invalid: 1")
		err = Zgemm(NoTrans, Trans, 2, 0, 0, alpha, a, 2, b, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("ldc invalid: 1")
		err = Zgemm(ConjTrans, NoTrans, 2, 0, 0, alpha, a, 1, b, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("ldc invalid: 1")
		err = Zgemm(ConjTrans, ConjTrans, 2, 0, 0, alpha, a, 1, b, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("ldc invalid: 1")
		err = Zgemm(ConjTrans, Trans, 2, 0, 0, alpha, a, 1, b, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("ldc invalid: 1")
		err = Zgemm(Trans, NoTrans, 2, 0, 0, alpha, a, 1, b, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("ldc invalid: 1")
		err = Zgemm(Trans, ConjTrans, 2, 0, 0, alpha, a, 1, b, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("ldc invalid: 1")
		err = Zgemm(Trans, Trans, 2, 0, 0, alpha, a, 1, b, 1, beta, c, 1)
		Chkxer(srnamt, err)
	case "Zhemm":
		*errt = fmt.Errorf("side invalid: /")
		err = Zhemm('/', Upper, 0, 0, alpha, a, 1, b, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("uplo invalid: /")
		err = Zhemm(Left, '/', 0, 0, alpha, a, 1, b, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("m invalid: -1")
		err = Zhemm(Left, Upper, -1, 0, alpha, a, 1, b, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("m invalid: -1")
		err = Zhemm(Right, Upper, -1, 0, alpha, a, 1, b, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("m invalid: -1")
		err = Zhemm(Left, Lower, -1, 0, alpha, a, 1, b, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("m invalid: -1")
		err = Zhemm(Right, Lower, -1, 0, alpha, a, 1, b, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("n invalid: -1")
		err = Zhemm(Left, Upper, 0, -1, alpha, a, 1, b, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("n invalid: -1")
		err = Zhemm(Right, Upper, 0, -1, alpha, a, 1, b, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("n invalid: -1")
		err = Zhemm(Left, Lower, 0, -1, alpha, a, 1, b, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("n invalid: -1")
		err = Zhemm(Right, Lower, 0, -1, alpha, a, 1, b, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("lda invalid: 1")
		err = Zhemm(Left, Upper, 2, 0, alpha, a, 1, b, 2, beta, c, 2)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("lda invalid: 1")
		err = Zhemm(Right, Upper, 0, 2, alpha, a, 1, b, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("lda invalid: 1")
		err = Zhemm(Left, Lower, 2, 0, alpha, a, 1, b, 2, beta, c, 2)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("lda invalid: 1")
		err = Zhemm(Right, Lower, 0, 2, alpha, a, 1, b, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("ldb invalid: 1")
		err = Zhemm(Left, Upper, 2, 0, alpha, a, 2, b, 1, beta, c, 2)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("ldb invalid: 1")
		err = Zhemm(Right, Upper, 2, 0, alpha, a, 1, b, 1, beta, c, 2)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("ldb invalid: 1")
		err = Zhemm(Left, Lower, 2, 0, alpha, a, 2, b, 1, beta, c, 2)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("ldb invalid: 1")
		err = Zhemm(Right, Lower, 2, 0, alpha, a, 1, b, 1, beta, c, 2)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("ldc invalid: 1")
		err = Zhemm(Left, Upper, 2, 0, alpha, a, 2, b, 2, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("ldc invalid: 1")
		err = Zhemm(Right, Upper, 2, 0, alpha, a, 1, b, 2, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("ldc invalid: 1")
		err = Zhemm(Left, Lower, 2, 0, alpha, a, 2, b, 2, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("ldc invalid: 1")
		err = Zhemm(Right, Lower, 2, 0, alpha, a, 1, b, 2, beta, c, 1)
		Chkxer(srnamt, err)
	case "Zsymm":
		*errt = fmt.Errorf("side invalid: /")
		err = Zsymm('/', Upper, 0, 0, alpha, a, 1, b, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("uplo invalid: /")
		err = Zsymm(Left, '/', 0, 0, alpha, a, 1, b, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("m invalid: -1")
		err = Zsymm(Left, Upper, -1, 0, alpha, a, 1, b, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("m invalid: -1")
		err = Zsymm(Right, Upper, -1, 0, alpha, a, 1, b, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("m invalid: -1")
		err = Zsymm(Left, Lower, -1, 0, alpha, a, 1, b, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("m invalid: -1")
		err = Zsymm(Right, Lower, -1, 0, alpha, a, 1, b, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("n invalid: -1")
		err = Zsymm(Left, Upper, 0, -1, alpha, a, 1, b, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("n invalid: -1")
		err = Zsymm(Right, Upper, 0, -1, alpha, a, 1, b, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("n invalid: -1")
		err = Zsymm(Left, Lower, 0, -1, alpha, a, 1, b, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("n invalid: -1")
		err = Zsymm(Right, Lower, 0, -1, alpha, a, 1, b, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("lda invalid: 1")
		err = Zsymm(Left, Upper, 2, 0, alpha, a, 1, b, 2, beta, c, 2)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("lda invalid: 1")
		err = Zsymm(Right, Upper, 0, 2, alpha, a, 1, b, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("lda invalid: 1")
		err = Zsymm(Left, Lower, 2, 0, alpha, a, 1, b, 2, beta, c, 2)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("lda invalid: 1")
		err = Zsymm(Right, Lower, 0, 2, alpha, a, 1, b, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("ldb invalid: 1")
		err = Zsymm(Left, Upper, 2, 0, alpha, a, 2, b, 1, beta, c, 2)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("ldb invalid: 1")
		err = Zsymm(Right, Upper, 2, 0, alpha, a, 1, b, 1, beta, c, 2)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("ldb invalid: 1")
		err = Zsymm(Left, Lower, 2, 0, alpha, a, 2, b, 1, beta, c, 2)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("ldb invalid: 1")
		err = Zsymm(Right, Lower, 2, 0, alpha, a, 1, b, 1, beta, c, 2)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("ldc invalid: 1")
		err = Zsymm(Left, Upper, 2, 0, alpha, a, 2, b, 2, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("ldc invalid: 1")
		err = Zsymm(Right, Upper, 2, 0, alpha, a, 1, b, 2, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("ldc invalid: 1")
		err = Zsymm(Left, Lower, 2, 0, alpha, a, 2, b, 2, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("ldc invalid: 1")
		err = Zsymm(Right, Lower, 2, 0, alpha, a, 1, b, 2, beta, c, 1)
		Chkxer(srnamt, err)
	case "Ztrmm":
		*errt = fmt.Errorf("side invalid: /")
		err = Ztrmm('/', Upper, NoTrans, NonUnit, 0, 0, alpha, a, 1, b, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("uplo invalid: /")
		err = Ztrmm(Left, '/', NoTrans, NonUnit, 0, 0, alpha, a, 1, b, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("transa invalid: /")
		err = Ztrmm(Left, Upper, '/', NonUnit, 0, 0, alpha, a, 1, b, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("diag invalid: /")
		err = Ztrmm(Left, Upper, NoTrans, '/', 0, 0, alpha, a, 1, b, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("m invalid: -1")
		err = Ztrmm(Left, Upper, NoTrans, NonUnit, -1, 0, alpha, a, 1, b, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("m invalid: -1")
		err = Ztrmm(Left, Upper, ConjTrans, NonUnit, -1, 0, alpha, a, 1, b, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("m invalid: -1")
		err = Ztrmm(Left, Upper, Trans, NonUnit, -1, 0, alpha, a, 1, b, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("m invalid: -1")
		err = Ztrmm(Right, Upper, NoTrans, NonUnit, -1, 0, alpha, a, 1, b, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("m invalid: -1")
		err = Ztrmm(Right, Upper, ConjTrans, NonUnit, -1, 0, alpha, a, 1, b, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("m invalid: -1")
		err = Ztrmm(Right, Upper, Trans, NonUnit, -1, 0, alpha, a, 1, b, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("m invalid: -1")
		err = Ztrmm(Left, Lower, NoTrans, NonUnit, -1, 0, alpha, a, 1, b, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("m invalid: -1")
		err = Ztrmm(Left, Lower, ConjTrans, NonUnit, -1, 0, alpha, a, 1, b, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("m invalid: -1")
		err = Ztrmm(Left, Lower, Trans, NonUnit, -1, 0, alpha, a, 1, b, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("m invalid: -1")
		err = Ztrmm(Right, Lower, NoTrans, NonUnit, -1, 0, alpha, a, 1, b, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("m invalid: -1")
		err = Ztrmm(Right, Lower, ConjTrans, NonUnit, -1, 0, alpha, a, 1, b, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("m invalid: -1")
		err = Ztrmm(Right, Lower, Trans, NonUnit, -1, 0, alpha, a, 1, b, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("n invalid: -1")
		err = Ztrmm(Left, Upper, NoTrans, NonUnit, 0, -1, alpha, a, 1, b, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("n invalid: -1")
		err = Ztrmm(Left, Upper, ConjTrans, NonUnit, 0, -1, alpha, a, 1, b, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("n invalid: -1")
		err = Ztrmm(Left, Upper, Trans, NonUnit, 0, -1, alpha, a, 1, b, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("n invalid: -1")
		err = Ztrmm(Right, Upper, NoTrans, NonUnit, 0, -1, alpha, a, 1, b, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("n invalid: -1")
		err = Ztrmm(Right, Upper, ConjTrans, NonUnit, 0, -1, alpha, a, 1, b, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("n invalid: -1")
		err = Ztrmm(Right, Upper, Trans, NonUnit, 0, -1, alpha, a, 1, b, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("n invalid: -1")
		err = Ztrmm(Left, Lower, NoTrans, NonUnit, 0, -1, alpha, a, 1, b, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("n invalid: -1")
		err = Ztrmm(Left, Lower, ConjTrans, NonUnit, 0, -1, alpha, a, 1, b, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("n invalid: -1")
		err = Ztrmm(Left, Lower, Trans, NonUnit, 0, -1, alpha, a, 1, b, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("n invalid: -1")
		err = Ztrmm(Right, Lower, NoTrans, NonUnit, 0, -1, alpha, a, 1, b, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("n invalid: -1")
		err = Ztrmm(Right, Lower, ConjTrans, NonUnit, 0, -1, alpha, a, 1, b, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("n invalid: -1")
		err = Ztrmm(Right, Lower, Trans, NonUnit, 0, -1, alpha, a, 1, b, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("lda invalid: 1")
		err = Ztrmm(Left, Upper, NoTrans, NonUnit, 2, 0, alpha, a, 1, b, 2)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("lda invalid: 1")
		err = Ztrmm(Left, Upper, ConjTrans, NonUnit, 2, 0, alpha, a, 1, b, 2)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("lda invalid: 1")
		err = Ztrmm(Left, Upper, Trans, NonUnit, 2, 0, alpha, a, 1, b, 2)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("lda invalid: 1")
		err = Ztrmm(Right, Upper, NoTrans, NonUnit, 0, 2, alpha, a, 1, b, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("lda invalid: 1")
		err = Ztrmm(Right, Upper, ConjTrans, NonUnit, 0, 2, alpha, a, 1, b, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("lda invalid: 1")
		err = Ztrmm(Right, Upper, Trans, NonUnit, 0, 2, alpha, a, 1, b, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("lda invalid: 1")
		err = Ztrmm(Left, Lower, NoTrans, NonUnit, 2, 0, alpha, a, 1, b, 2)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("lda invalid: 1")
		err = Ztrmm(Left, Lower, ConjTrans, NonUnit, 2, 0, alpha, a, 1, b, 2)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("lda invalid: 1")
		err = Ztrmm(Left, Lower, Trans, NonUnit, 2, 0, alpha, a, 1, b, 2)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("lda invalid: 1")
		err = Ztrmm(Right, Lower, NoTrans, NonUnit, 0, 2, alpha, a, 1, b, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("lda invalid: 1")
		err = Ztrmm(Right, Lower, ConjTrans, NonUnit, 0, 2, alpha, a, 1, b, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("lda invalid: 1")
		err = Ztrmm(Right, Lower, Trans, NonUnit, 0, 2, alpha, a, 1, b, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("ldb invalid: 1")
		err = Ztrmm(Left, Upper, NoTrans, NonUnit, 2, 0, alpha, a, 2, b, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("ldb invalid: 1")
		err = Ztrmm(Left, Upper, ConjTrans, NonUnit, 2, 0, alpha, a, 2, b, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("ldb invalid: 1")
		err = Ztrmm(Left, Upper, Trans, NonUnit, 2, 0, alpha, a, 2, b, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("ldb invalid: 1")
		err = Ztrmm(Right, Upper, NoTrans, NonUnit, 2, 0, alpha, a, 1, b, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("ldb invalid: 1")
		err = Ztrmm(Right, Upper, ConjTrans, NonUnit, 2, 0, alpha, a, 1, b, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("ldb invalid: 1")
		err = Ztrmm(Right, Upper, Trans, NonUnit, 2, 0, alpha, a, 1, b, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("ldb invalid: 1")
		err = Ztrmm(Left, Lower, NoTrans, NonUnit, 2, 0, alpha, a, 2, b, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("ldb invalid: 1")
		err = Ztrmm(Left, Lower, ConjTrans, NonUnit, 2, 0, alpha, a, 2, b, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("ldb invalid: 1")
		err = Ztrmm(Left, Lower, Trans, NonUnit, 2, 0, alpha, a, 2, b, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("ldb invalid: 1")
		err = Ztrmm(Right, Lower, NoTrans, NonUnit, 2, 0, alpha, a, 1, b, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("ldb invalid: 1")
		err = Ztrmm(Right, Lower, ConjTrans, NonUnit, 2, 0, alpha, a, 1, b, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("ldb invalid: 1")
		err = Ztrmm(Right, Lower, Trans, NonUnit, 2, 0, alpha, a, 1, b, 1)
		Chkxer(srnamt, err)
	case "Ztrsm":
		*errt = fmt.Errorf("side invalid: /")
		err = Ztrsm('/', Upper, NoTrans, NonUnit, 0, 0, alpha, a, 1, b, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("uplo invalid: /")
		err = Ztrsm(Left, '/', NoTrans, NonUnit, 0, 0, alpha, a, 1, b, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("transa invalid: /")
		err = Ztrsm(Left, Upper, '/', NonUnit, 0, 0, alpha, a, 1, b, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("diag invalid: /")
		err = Ztrsm(Left, Upper, NoTrans, '/', 0, 0, alpha, a, 1, b, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("m invalid: -1")
		err = Ztrsm(Left, Upper, NoTrans, NonUnit, -1, 0, alpha, a, 1, b, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("m invalid: -1")
		err = Ztrsm(Left, Upper, ConjTrans, NonUnit, -1, 0, alpha, a, 1, b, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("m invalid: -1")
		err = Ztrsm(Left, Upper, Trans, NonUnit, -1, 0, alpha, a, 1, b, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("m invalid: -1")
		err = Ztrsm(Right, Upper, NoTrans, NonUnit, -1, 0, alpha, a, 1, b, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("m invalid: -1")
		err = Ztrsm(Right, Upper, ConjTrans, NonUnit, -1, 0, alpha, a, 1, b, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("m invalid: -1")
		err = Ztrsm(Right, Upper, Trans, NonUnit, -1, 0, alpha, a, 1, b, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("m invalid: -1")
		err = Ztrsm(Left, Lower, NoTrans, NonUnit, -1, 0, alpha, a, 1, b, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("m invalid: -1")
		err = Ztrsm(Left, Lower, ConjTrans, NonUnit, -1, 0, alpha, a, 1, b, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("m invalid: -1")
		err = Ztrsm(Left, Lower, Trans, NonUnit, -1, 0, alpha, a, 1, b, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("m invalid: -1")
		err = Ztrsm(Right, Lower, NoTrans, NonUnit, -1, 0, alpha, a, 1, b, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("m invalid: -1")
		err = Ztrsm(Right, Lower, ConjTrans, NonUnit, -1, 0, alpha, a, 1, b, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("m invalid: -1")
		err = Ztrsm(Right, Lower, Trans, NonUnit, -1, 0, alpha, a, 1, b, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("n invalid: -1")
		err = Ztrsm(Left, Upper, NoTrans, NonUnit, 0, -1, alpha, a, 1, b, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("n invalid: -1")
		err = Ztrsm(Left, Upper, ConjTrans, NonUnit, 0, -1, alpha, a, 1, b, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("n invalid: -1")
		err = Ztrsm(Left, Upper, Trans, NonUnit, 0, -1, alpha, a, 1, b, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("n invalid: -1")
		err = Ztrsm(Right, Upper, NoTrans, NonUnit, 0, -1, alpha, a, 1, b, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("n invalid: -1")
		err = Ztrsm(Right, Upper, ConjTrans, NonUnit, 0, -1, alpha, a, 1, b, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("n invalid: -1")
		err = Ztrsm(Right, Upper, Trans, NonUnit, 0, -1, alpha, a, 1, b, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("n invalid: -1")
		err = Ztrsm(Left, Lower, NoTrans, NonUnit, 0, -1, alpha, a, 1, b, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("n invalid: -1")
		err = Ztrsm(Left, Lower, ConjTrans, NonUnit, 0, -1, alpha, a, 1, b, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("n invalid: -1")
		err = Ztrsm(Left, Lower, Trans, NonUnit, 0, -1, alpha, a, 1, b, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("n invalid: -1")
		err = Ztrsm(Right, Lower, NoTrans, NonUnit, 0, -1, alpha, a, 1, b, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("n invalid: -1")
		err = Ztrsm(Right, Lower, ConjTrans, NonUnit, 0, -1, alpha, a, 1, b, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("n invalid: -1")
		err = Ztrsm(Right, Lower, Trans, NonUnit, 0, -1, alpha, a, 1, b, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("lda invalid: 1")
		err = Ztrsm(Left, Upper, NoTrans, NonUnit, 2, 0, alpha, a, 1, b, 2)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("lda invalid: 1")
		err = Ztrsm(Left, Upper, ConjTrans, NonUnit, 2, 0, alpha, a, 1, b, 2)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("lda invalid: 1")
		err = Ztrsm(Left, Upper, Trans, NonUnit, 2, 0, alpha, a, 1, b, 2)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("lda invalid: 1")
		err = Ztrsm(Right, Upper, NoTrans, NonUnit, 0, 2, alpha, a, 1, b, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("lda invalid: 1")
		err = Ztrsm(Right, Upper, ConjTrans, NonUnit, 0, 2, alpha, a, 1, b, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("lda invalid: 1")
		err = Ztrsm(Right, Upper, Trans, NonUnit, 0, 2, alpha, a, 1, b, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("lda invalid: 1")
		err = Ztrsm(Left, Lower, NoTrans, NonUnit, 2, 0, alpha, a, 1, b, 2)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("lda invalid: 1")
		err = Ztrsm(Left, Lower, ConjTrans, NonUnit, 2, 0, alpha, a, 1, b, 2)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("lda invalid: 1")
		err = Ztrsm(Left, Lower, Trans, NonUnit, 2, 0, alpha, a, 1, b, 2)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("lda invalid: 1")
		err = Ztrsm(Right, Lower, NoTrans, NonUnit, 0, 2, alpha, a, 1, b, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("lda invalid: 1")
		err = Ztrsm(Right, Lower, ConjTrans, NonUnit, 0, 2, alpha, a, 1, b, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("lda invalid: 1")
		err = Ztrsm(Right, Lower, Trans, NonUnit, 0, 2, alpha, a, 1, b, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("ldb invalid: 1")
		err = Ztrsm(Left, Upper, NoTrans, NonUnit, 2, 0, alpha, a, 2, b, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("ldb invalid: 1")
		err = Ztrsm(Left, Upper, ConjTrans, NonUnit, 2, 0, alpha, a, 2, b, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("ldb invalid: 1")
		err = Ztrsm(Left, Upper, Trans, NonUnit, 2, 0, alpha, a, 2, b, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("ldb invalid: 1")
		err = Ztrsm(Right, Upper, NoTrans, NonUnit, 2, 0, alpha, a, 1, b, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("ldb invalid: 1")
		err = Ztrsm(Right, Upper, ConjTrans, NonUnit, 2, 0, alpha, a, 1, b, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("ldb invalid: 1")
		err = Ztrsm(Right, Upper, Trans, NonUnit, 2, 0, alpha, a, 1, b, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("ldb invalid: 1")
		err = Ztrsm(Left, Lower, NoTrans, NonUnit, 2, 0, alpha, a, 2, b, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("ldb invalid: 1")
		err = Ztrsm(Left, Lower, ConjTrans, NonUnit, 2, 0, alpha, a, 2, b, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("ldb invalid: 1")
		err = Ztrsm(Left, Lower, Trans, NonUnit, 2, 0, alpha, a, 2, b, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("ldb invalid: 1")
		err = Ztrsm(Right, Lower, NoTrans, NonUnit, 2, 0, alpha, a, 1, b, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("ldb invalid: 1")
		err = Ztrsm(Right, Lower, ConjTrans, NonUnit, 2, 0, alpha, a, 1, b, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("ldb invalid: 1")
		err = Ztrsm(Right, Lower, Trans, NonUnit, 2, 0, alpha, a, 1, b, 1)
		Chkxer(srnamt, err)
	case "Zherk":
		*errt = fmt.Errorf("uplo invalid: /")
		err = Zherk('/', NoTrans, 0, 0, ralpha, a, 1, rbeta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("trans invalid: Trans")
		err = Zherk(Upper, Trans, 0, 0, ralpha, a, 1, rbeta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("n invalid: -1")
		err = Zherk(Upper, NoTrans, -1, 0, ralpha, a, 1, rbeta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("n invalid: -1")
		err = Zherk(Upper, ConjTrans, -1, 0, ralpha, a, 1, rbeta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("n invalid: -1")
		err = Zherk(Lower, NoTrans, -1, 0, ralpha, a, 1, rbeta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("n invalid: -1")
		err = Zherk(Lower, ConjTrans, -1, 0, ralpha, a, 1, rbeta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("k invalid: -1")
		err = Zherk(Upper, NoTrans, 0, -1, ralpha, a, 1, rbeta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("k invalid: -1")
		err = Zherk(Upper, ConjTrans, 0, -1, ralpha, a, 1, rbeta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("k invalid: -1")
		err = Zherk(Lower, NoTrans, 0, -1, ralpha, a, 1, rbeta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("k invalid: -1")
		err = Zherk(Lower, ConjTrans, 0, -1, ralpha, a, 1, rbeta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("lda invalid: 1")
		err = Zherk(Upper, NoTrans, 2, 0, ralpha, a, 1, rbeta, c, 2)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("lda invalid: 1")
		err = Zherk(Upper, ConjTrans, 0, 2, ralpha, a, 1, rbeta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("lda invalid: 1")
		err = Zherk(Lower, NoTrans, 2, 0, ralpha, a, 1, rbeta, c, 2)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("lda invalid: 1")
		err = Zherk(Lower, ConjTrans, 0, 2, ralpha, a, 1, rbeta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("ldc invalid: 1")
		err = Zherk(Upper, NoTrans, 2, 0, ralpha, a, 2, rbeta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("ldc invalid: 1")
		err = Zherk(Upper, ConjTrans, 2, 0, ralpha, a, 1, rbeta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("ldc invalid: 1")
		err = Zherk(Lower, NoTrans, 2, 0, ralpha, a, 2, rbeta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("ldc invalid: 1")
		err = Zherk(Lower, ConjTrans, 2, 0, ralpha, a, 1, rbeta, c, 1)
		Chkxer(srnamt, err)
	case "Zsyrk":
		*errt = fmt.Errorf("uplo invalid: /")
		err = Zsyrk('/', NoTrans, 0, 0, alpha, a, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("trans invalid: ConjTrans")
		err = Zsyrk(Upper, ConjTrans, 0, 0, alpha, a, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("n invalid: -1")
		err = Zsyrk(Upper, NoTrans, -1, 0, alpha, a, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("n invalid: -1")
		err = Zsyrk(Upper, Trans, -1, 0, alpha, a, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("n invalid: -1")
		err = Zsyrk(Lower, NoTrans, -1, 0, alpha, a, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("n invalid: -1")
		err = Zsyrk(Lower, Trans, -1, 0, alpha, a, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("k invalid: -1")
		err = Zsyrk(Upper, NoTrans, 0, -1, alpha, a, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("k invalid: -1")
		err = Zsyrk(Upper, Trans, 0, -1, alpha, a, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("k invalid: -1")
		err = Zsyrk(Lower, NoTrans, 0, -1, alpha, a, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("k invalid: -1")
		err = Zsyrk(Lower, Trans, 0, -1, alpha, a, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("lda invalid: 1")
		err = Zsyrk(Upper, NoTrans, 2, 0, alpha, a, 1, beta, c, 2)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("lda invalid: 1")
		err = Zsyrk(Upper, Trans, 0, 2, alpha, a, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("lda invalid: 1")
		err = Zsyrk(Lower, NoTrans, 2, 0, alpha, a, 1, beta, c, 2)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("lda invalid: 1")
		err = Zsyrk(Lower, Trans, 0, 2, alpha, a, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("ldc invalid: 1")
		err = Zsyrk(Upper, NoTrans, 2, 0, alpha, a, 2, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("ldc invalid: 1")
		err = Zsyrk(Upper, Trans, 2, 0, alpha, a, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("ldc invalid: 1")
		err = Zsyrk(Lower, NoTrans, 2, 0, alpha, a, 2, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("ldc invalid: 1")
		err = Zsyrk(Lower, Trans, 2, 0, alpha, a, 1, beta, c, 1)
		Chkxer(srnamt, err)
	case "Zher2k":
		*errt = fmt.Errorf("uplo invalid: /")
		err = Zher2k('/', NoTrans, 0, 0, alpha, a, 1, b, 1, rbeta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("trans invalid: Trans")
		err = Zher2k(Upper, Trans, 0, 0, alpha, a, 1, b, 1, rbeta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("n invalid: -1")
		err = Zher2k(Upper, NoTrans, -1, 0, alpha, a, 1, b, 1, rbeta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("n invalid: -1")
		err = Zher2k(Upper, ConjTrans, -1, 0, alpha, a, 1, b, 1, rbeta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("n invalid: -1")
		err = Zher2k(Lower, NoTrans, -1, 0, alpha, a, 1, b, 1, rbeta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("n invalid: -1")
		err = Zher2k(Lower, ConjTrans, -1, 0, alpha, a, 1, b, 1, rbeta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("k invalid: -1")
		err = Zher2k(Upper, NoTrans, 0, -1, alpha, a, 1, b, 1, rbeta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("k invalid: -1")
		err = Zher2k(Upper, ConjTrans, 0, -1, alpha, a, 1, b, 1, rbeta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("k invalid: -1")
		err = Zher2k(Lower, NoTrans, 0, -1, alpha, a, 1, b, 1, rbeta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("k invalid: -1")
		err = Zher2k(Lower, ConjTrans, 0, -1, alpha, a, 1, b, 1, rbeta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("lda invalid: 1")
		err = Zher2k(Upper, NoTrans, 2, 0, alpha, a, 1, b, 1, rbeta, c, 2)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("lda invalid: 1")
		err = Zher2k(Upper, ConjTrans, 0, 2, alpha, a, 1, b, 1, rbeta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("lda invalid: 1")
		err = Zher2k(Lower, NoTrans, 2, 0, alpha, a, 1, b, 1, rbeta, c, 2)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("lda invalid: 1")
		err = Zher2k(Lower, ConjTrans, 0, 2, alpha, a, 1, b, 1, rbeta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("ldb invalid: 1")
		err = Zher2k(Upper, NoTrans, 2, 0, alpha, a, 2, b, 1, rbeta, c, 2)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("ldb invalid: 1")
		err = Zher2k(Upper, ConjTrans, 0, 2, alpha, a, 2, b, 1, rbeta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("ldb invalid: 1")
		err = Zher2k(Lower, NoTrans, 2, 0, alpha, a, 2, b, 1, rbeta, c, 2)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("ldb invalid: 1")
		err = Zher2k(Lower, ConjTrans, 0, 2, alpha, a, 2, b, 1, rbeta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("ldc invalid: 1")
		err = Zher2k(Upper, NoTrans, 2, 0, alpha, a, 2, b, 2, rbeta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("ldc invalid: 1")
		err = Zher2k(Upper, ConjTrans, 2, 0, alpha, a, 1, b, 1, rbeta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("ldc invalid: 1")
		err = Zher2k(Lower, NoTrans, 2, 0, alpha, a, 2, b, 2, rbeta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("ldc invalid: 1")
		err = Zher2k(Lower, ConjTrans, 2, 0, alpha, a, 1, b, 1, rbeta, c, 1)
		Chkxer(srnamt, err)
	case "Zsyr2k":
		*errt = fmt.Errorf("uplo invalid: /")
		err = Zsyr2k('/', NoTrans, 0, 0, alpha, a, 1, b, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("trans invalid: ConjTrans")
		err = Zsyr2k(Upper, ConjTrans, 0, 0, alpha, a, 1, b, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("n invalid: -1")
		err = Zsyr2k(Upper, NoTrans, -1, 0, alpha, a, 1, b, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("n invalid: -1")
		err = Zsyr2k(Upper, Trans, -1, 0, alpha, a, 1, b, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("n invalid: -1")
		err = Zsyr2k(Lower, NoTrans, -1, 0, alpha, a, 1, b, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("n invalid: -1")
		err = Zsyr2k(Lower, Trans, -1, 0, alpha, a, 1, b, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("k invalid: -1")
		err = Zsyr2k(Upper, NoTrans, 0, -1, alpha, a, 1, b, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("k invalid: -1")
		err = Zsyr2k(Upper, Trans, 0, -1, alpha, a, 1, b, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("k invalid: -1")
		err = Zsyr2k(Lower, NoTrans, 0, -1, alpha, a, 1, b, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("k invalid: -1")
		err = Zsyr2k(Lower, Trans, 0, -1, alpha, a, 1, b, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("lda invalid: 1")
		err = Zsyr2k(Upper, NoTrans, 2, 0, alpha, a, 1, b, 1, beta, c, 2)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("lda invalid: 1")
		err = Zsyr2k(Upper, Trans, 0, 2, alpha, a, 1, b, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("lda invalid: 1")
		err = Zsyr2k(Lower, NoTrans, 2, 0, alpha, a, 1, b, 1, beta, c, 2)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("lda invalid: 1")
		err = Zsyr2k(Lower, Trans, 0, 2, alpha, a, 1, b, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("ldb invalid: 1")
		err = Zsyr2k(Upper, NoTrans, 2, 0, alpha, a, 2, b, 1, beta, c, 2)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("ldb invalid: 1")
		err = Zsyr2k(Upper, Trans, 0, 2, alpha, a, 2, b, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("ldb invalid: 1")
		err = Zsyr2k(Lower, NoTrans, 2, 0, alpha, a, 2, b, 1, beta, c, 2)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("ldb invalid: 1")
		err = Zsyr2k(Lower, Trans, 0, 2, alpha, a, 2, b, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("ldc invalid: 1")
		err = Zsyr2k(Upper, NoTrans, 2, 0, alpha, a, 2, b, 2, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("ldc invalid: 1")
		err = Zsyr2k(Upper, Trans, 2, 0, alpha, a, 1, b, 1, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("ldc invalid: 1")
		err = Zsyr2k(Lower, NoTrans, 2, 0, alpha, a, 2, b, 2, beta, c, 1)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("ldc invalid: 1")
		err = Zsyr2k(Lower, Trans, 2, 0, alpha, a, 1, b, 1, beta, c, 1)
		Chkxer(srnamt, err)
	}

	if *ok {
		fmt.Printf(" %6s passed the tests of error-exits\n", srnamt)
	} else {
		fmt.Printf(" ******* %6s FAILED THE TESTS OF ERROR-EXITS *******\n", srnamt)
	}

	return
}
